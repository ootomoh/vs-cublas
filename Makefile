NVCC=nvcc
COMMON_SOURCE=main.cu
COMMON_FLAG=-std=c++11
CUBLAS_FLAG=-lcublas

exec_cublasSgemv: $(CUBLAS_SOURCE) $(COMMON_SOURCE)
	$(NVCC) $(COMMON_FLAG) $(CUBLAS_FLAG) $(COMMON_SOURCE) -o $@ -DUSE_CUBLAS

clean:
	rm -rf exec*
