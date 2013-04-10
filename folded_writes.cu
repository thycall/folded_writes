#include <vector>
#include <assert.h>

enum { FOLD_LENGTH = 32, 
       NUM_PASSES  = 128, 
       BDIM        = NUM_PASSES 
};

/* We will write the slot into the upper 8 bits. */

#define RB_BIT  24

/* Mask for the lower 24 bits. */

#define RB_MASK 0xffffff

/* Convert a 2D index (i,j) into a 
   1D folded index. The folded data format is a hybrid 
   between SoA and AoS;

   SoA: | a0 a1 a2 a3 a4 a5 a6 a7 b0 b1 b2 b3 b4 b5 b6 b7 | 
   AoS: | a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7 |
   F-4: | a0 a1 a2 a3 b0 b1 b2 b3 a4 a5 a6 a7 b4 b5 b6 b7 |

   where F-4 is the folded format with FOLD_LENGTH=4 */   

#define FOLD_INDEX(i,icomp,ncomp)               \
  ((i)&(FOLD_LENGTH-1))+                        \
    ((icomp)*FOLD_LENGTH) +                     \
    ((i)&(~(FOLD_LENGTH-1)))*(ncomp)

/* Round up x so that it is a multiple of FOLD_LENGTH */

#define FOLD_LENGTH_PAD(x)                      \
  (((x)+FOLD_LENGTH-1)&(~(FOLD_LENGTH-1)))


/* Compare the checksums in _result0 and _result1, if they 
   do not match atomically increment _num_failures by one.
   The length of the arrays _result[01] is NUM_PASSES, 
   and the block dim of this kernel is BDIM. So each 
   thread gets to compare one element of the array. Should
   have a BOOST_STATIC_ASSERT( BDIM == NUM_PASSES ); */

static __global__ void
COMPARE( uint32_t const * _result0, 
         uint32_t const * _result1,
         uint32_t       * _num_failures )
{
  uint32_t r0 = _result0[threadIdx.x];
  uint32_t r1 = _result1[threadIdx.x];
  
  if( r0 != r1 )
    atomicInc(_num_failures,0xffffffff);
}

/* Hash the data in result (array of length N) and write it 
   into _checksum. 
   To do the checksum we interpret each component of each 
   float4 as a uint32_t and rotate and XOR the components 
   together. We then do a warp-wide XOR all-reduce, and the
   first thread in the warp XORs the result atomically into 
   the global memory location _checksum. */

static __global__ void
HASH( float4   const * _result, 
      uint32_t       * _checksum, 
      int              N )
{
  int idx = threadIdx.x + BDIM * blockIdx.x;

  int Npad = (N + warpSize - 1)&~(warpSize-1);

  /* We do not really need to do this padding, but somebody
     could object that the warp-wide shuffle below may not 
     be defined if some of the theoretically participating 
     threads have been retired. */

  if( idx >= Npad ) return;

  float4   result = idx < N ? _result[idx] :
    make_float4(0.f,0.f,0.f,0.f);

  int32_t local  = 0;
  {
    uint32_t & a =  reinterpret_cast<uint32_t&>(result.x);
    uint32_t & b =  reinterpret_cast<uint32_t&>(result.y);
    uint32_t & c =  reinterpret_cast<uint32_t&>(result.z);
    uint32_t & d =  reinterpret_cast<uint32_t&>(result.w);
    local ^= (a>>1)|(a<<(32-1));
    local ^= (b>>2)|(b<<(32-2));
    local ^= (c>>3)|(c<<(32-3));
    local ^= (d>>4)|(d<<(32-4));
  }
  
  local ^= __shfl_xor(local,  1, warpSize);
  local ^= __shfl_xor(local,  2, warpSize);
  local ^= __shfl_xor(local,  4, warpSize);
  local ^= __shfl_xor(local,  8, warpSize);
  local ^= __shfl_xor(local, 16, warpSize);

  if( (threadIdx.x & 0x1f) == 0 )
    atomicXor( _checksum, local );
}

/* Do the actual work:

   do in parallel:
      read two indices from a folded index array, 
      determine the slot to scatter to, and the 
      particle index. 
      compute some arbitrary float4, 
      and scatter this data to the appropriate slot. */

static __global__ void 
SCATTER_FOLDED( float4       * _dst, 
                uint         * _indices, 
                float          delta, 
                int            M )
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  if( idx >= M )  return;

  uint i0 = _indices[FOLD_INDEX(idx,0,2)];
  uint i1 = _indices[FOLD_INDEX(idx,1,2)];
  
  /* Extract the slot */
  uint slot0 = i0 >> RB_BIT;
  uint slot1 = i1 >> RB_BIT;

  /* Determine the index by masking off the slot info */
  i0 &= RB_MASK;
  i1 &= RB_MASK;

  /* Create some arbitrary non-constant float4. */
  float4 d = make_float4(0.f,0.f,0.f,0.f);
  d.x = 1.0e-6f * delta * idx;
  d.y = 1.1e-6f * delta * idx;
  d.z = 1.2e-6f * delta * idx;
  d.w = 1.3e-6f * delta * idx;
  
  float4 minusd = make_float4(-d.x,-d.y,-d.z,-d.w);

  /* Scatter it. */
  _dst[FOLD_INDEX(i0,slot0,2)] =      d;
  _dst[FOLD_INDEX(i1,slot1,2)] = minusd;
}


/* This simulates the write transactions encountered when 
   calculating binary, bonded force-terms, like a stretch 
   term. 
   
   As a model we will use something like water, in the 
   sense that it has two stretch terms that share one 
   node. 

   In order to be able to calculate all the terms 
   concurrently we must make sure that the terms sharing 
   a node do not enter a write-conflict when scattering 
   the "force" associated with the shared node. We therefore
   create a force buffer with two components, so that one 
   term writes the contribution to the shared node into the
   first component and the other term writes it into the 
   second component. */

struct FoldedWrites 
{
  /* Scalar to store the total number of failures 
     encountered. Atomically incremented. */
  uint32_t          * dev_num_failures;
  /* Store reference checksums, array of length NUM_PASSES. */
  uint32_t          * dev_reference_checksum;
  /* Store trial checksums, array of length NUM_PASSES. */
  uint32_t          * dev_checksums;
  /* Data-buffer we are going to scatter into */
  float4            * dev_data;
  /* Scatter indices. */
  uint32_t          * dev_indices;

  int nTerms, nParticles;
  
  FoldedWrites( int nWaters )
    : dev_num_failures(), dev_reference_checksum(), 
      dev_checksums(), dev_data(), dev_indices(),
      nTerms(2*nWaters), nParticles(3*nWaters)
  {
    std::vector<int> particleOrder(nParticles);
    for( size_t i = 0; i < particleOrder.size(); ++i )
      particleOrder[i] = i;
    
    /* Mix them up just a little bit, random quality 
       does not matter here. */

    double swap_rate  = 0.2;
    size_t swap_width = 6;
    for( size_t i = 0; i < particleOrder.size()-swap_width; ++i ){
      double coin = (double)rand()/(double)RAND_MAX;
      if( coin < swap_rate ){
        int delta = rand() % swap_width;
        int index = particleOrder[i+delta];
        particleOrder[i+delta] = particleOrder[i];
        particleOrder[i      ] = index;
      }
    }
    
    /* Build the index list */
    std::vector<uint32_t> particleIndices( FOLD_LENGTH_PAD(nTerms) * 2 );
    for( size_t i = 0; i < nWaters; ++i ){
      int O  = particleOrder[3*i+0];
      int H1 = particleOrder[3*i+1];
      int H2 = particleOrder[3*i+2];
      assert( O < nParticles && O >= 0 );
      assert( H2 < nParticles && H2 >= 0  );
      assert( H1 < nParticles && H1 >= 0 );
      particleIndices[FOLD_INDEX(2*i  ,0,2)] = O  | (0<<RB_BIT);
      particleIndices[FOLD_INDEX(2*i  ,1,2)] = H1 | (0<<RB_BIT);
      particleIndices[FOLD_INDEX(2*i+1,0,2)] = O  | (1<<RB_BIT);
      particleIndices[FOLD_INDEX(2*i+1,1,2)] = H2 | (0<<RB_BIT);
    }

    /* Demonstrate that there are no conflicts */
    {
      std::vector<bool> slot_used(nParticles*2);
      for( size_t i = 0; i < nTerms; ++i ) {
        uint32_t p0 = particleIndices[FOLD_INDEX(i,0,2)];
        uint32_t p1 = particleIndices[FOLD_INDEX(i,1,2)];
        
        uint32_t slot0 = p0 >> RB_BIT;
        uint32_t slot1 = p1 >> RB_BIT;
        
        p0 &= RB_MASK;
        p1 &= RB_MASK;

        if( slot_used[2*p0+slot0] ){
          fprintf(stderr,"Index set will lead to write conflicts\n");
          exit(1);
        } else {
          slot_used[2*p0+slot0] = true;
        }
        
        if( slot_used[2*p1+slot1] ){
          fprintf(stderr,"Index set will lead to write conflicts\n");
          exit(1);
        } else {
          slot_used[2*p1+slot1] = true;
        }
      }
    }
    
    /* Allocate device buffer for indices and upload them */
    cudaMalloc(&dev_indices, sizeof(uint32_t) * FOLD_LENGTH_PAD( nTerms ) * 2);
    cudaMemcpy(dev_indices, particleIndices.data(), 
               sizeof(uint32_t) * FOLD_LENGTH_PAD( nTerms ) * 2, 
               cudaMemcpyHostToDevice);
    
    /* Allocate space for the destination buffer */
    cudaMalloc(&dev_data, sizeof(float4) * FOLD_LENGTH_PAD(nParticles) * 2);
    /* Clear it just to be nice. */
    cudaMemset(dev_data, 0x0, sizeof(float4) * FOLD_LENGTH_PAD(nParticles) * 2);

    /* Allocate auxilliary data */
    cudaMalloc( &dev_reference_checksum, sizeof(uint32_t) * NUM_PASSES );
    cudaMalloc( &dev_checksums, sizeof(uint32_t) * NUM_PASSES );
    cudaMalloc( &dev_num_failures, sizeof(uint32_t) );
    cudaMemset( dev_num_failures, 0x0, sizeof(uint32_t) );
  }
  
  ~FoldedWrites()
  {
    cudaFree(dev_num_failures);
    cudaFree(dev_reference_checksum);
    cudaFree(dev_checksums);
    cudaFree(dev_data);
    cudaFree(dev_indices);
    cudaDeviceReset();
  }
  
  /* Do NUM_PASSES passes of writes, with pass i checksumming the 
     result and writing into dev_checksum_write_buffer[i]. */
  
  void doPass( cudaStream_t const & stream, 
               uint32_t           * dev_checksum_write_buffer )
  {
    cudaMemsetAsync( dev_checksum_write_buffer, 0x0, 
                     sizeof(uint32_t) * NUM_PASSES, stream );
    
    for( size_t i = 0; i < NUM_PASSES; ++i ){
      int nBlocksS = (nTerms + BDIM - 1)/BDIM;
      SCATTER_FOLDED<<< nBlocksS, BDIM, 0, stream >>>
        ( dev_data, 
          dev_indices, 
          i * 1.234e-3f,
          nTerms );
      int nBlocksP = (nParticles + BDIM - 1)/BDIM;
      HASH<<< nBlocksP, BDIM, 0, stream >>>
        ( dev_data, dev_checksum_write_buffer + i, nParticles );
    }
  }

  /* Compare the two arrays dev0, and dev1 and for each 
     miscompare, increment dev_num_failures by one. */

  void check( cudaStream_t const & stream, 
              uint32_t     const * dev0, 
              uint32_t     const * dev1 )
  {
    int nBlocks = 1;
    COMPARE<<<nBlocks, BDIM, 0, stream>>>
      ( dev0, dev1, dev_num_failures );
  }

  
  int doCheck( size_t passes )
  {
    cudaStream_t stream;
    cudaStreamCreate( &stream );

    doPass( stream, dev_reference_checksum );
    
    for( size_t i = 0; i < passes; ++i ){
      doPass( stream, dev_checksums );
      check( stream, dev_checksums, dev_reference_checksum );
    }

    /* Download the number of failures to the host and 
       return it. */
    uint32_t num_failures[1];
    cudaMemcpy( num_failures, dev_num_failures,
                sizeof(uint32_t), cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize();
    
    cudaError_t lastError = cudaGetLastError();
    if( cudaSuccess != lastError ){
      fprintf(stderr,"CUDA ERROR: %s\n",
              cudaGetErrorString(lastError));
      exit(1);
    }
    
    cudaStreamDestroy( stream );
    return *num_failures;
  }
};
  
    
    

int main( int argc, char * argv[] )
{
  if( argc < 2 ) {
    printf("Usage: %s [num_passes]\n",argv[0]);
    printf("Takes about 3 minutes per 100 passes\n");
    return 1;
  }

  int num_passes = atoi(argv[1]);
  printf("Going to run %8d passes\n",num_passes);
  fflush(stdout);

  FoldedWrites memcheck( 4234567 );
  
  int num_failures = -1;
  for( size_t i = 0; i < num_passes; ++i ){
    num_failures = memcheck.doCheck( 128 );
    if( num_failures ){
      fprintf(stderr,"Detected %d failures\n",num_failures);
      printf("Detected %d failures\n",num_failures);
      exit(1);
    }
    if( i%1 == 0 ){
      printf("Pass %8d complete\n",i);
      fflush(stdout);
    }
  }
  if( num_failures == 0 )
    printf("OK\n");
  return 0;
}

  


  
  
  
