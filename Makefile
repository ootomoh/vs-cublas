NVCC=nvcc
MYSGEMV_SOURCE=mySgemv.cu
COMMON_SOURCE=main.cu
COMMON_FLAG=-std=c++11 -lboost_program_options -Wno-deprecated-gpu-targets 
CUBLAS_FLAG=-lcublas

exec_multitest: $(COMMON_SOURCE) $(MYSGEMV_SOURCE)
	$(NVCC) $(COMMON_FLAG) $(CUBLAS_FLAG) $(COMMON_SOURCE) $(MYSGEMV_SOURCE) -o $@


clean:
	rm -rf exec*
