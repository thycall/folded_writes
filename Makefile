all:
	nvcc  -use_fast_math -o foldedWrites folded_writes.cu -O3	\
		-gencode arch=compute_30,code=\"sm_30,compute_30\"  \
	  -Xptxas -v

