all:
	nvcc  -use_fast_math -o foldedWrites folded_writes.cu -O3	\
		-gencode arch=compute_30,code=\"sm_30,compute_30\"  \
		-gencode arch=compute_20,code=\"sm_20,compute_20\"  \
	  -Xptxas -v

