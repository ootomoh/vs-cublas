#include <iostream>
#include <random>

// matrix : N x N
// vector : N
const int N = 1<<12;
// how many calculations
const int CALC = 1<<10;

#ifdef USE_CUBLAS
#include <cublas.h>
#include <cublas_v2.h>
#define CUBLAS_HANDLE_ERROR( err ) (cublasStatusError( err, __FILE__, __LINE__ ))
const char* cublasGetErrorString(cublasStatus_t status)
{
	switch(status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
	default: return "unknown error";
	}
}
static void cublasStatusError( cublasStatus_t err,
		const char *file,
		int line ) {
	if (err != CUBLAS_STATUS_SUCCESS) {
		printf( "cuBLAS Error:\n%s in %s at line %d\n", cublasGetErrorString(err),
				file, line );
		exit( EXIT_FAILURE );
	}
}
#else
void mySgemv(int m,int n,const float *A,const float* x,float *y);
#endif

#define CUDA_HANDLE_ERROR( err ) (cudaHandleError( err, __FILE__, __LINE__ ))
static void cudaHandleError( cudaError_t err,
		const char *file,
		int line ) {
	if (err != cudaSuccess) {
		printf( "CUDA Error\n%s in %s at line %d\n", cudaGetErrorString( err ),
				file, line );
		exit( EXIT_FAILURE );
	}
}



int initMatrixAndVector(
		float **host_matrix,
		float **device_matrix,
		float **host_vector,
		float **device_vector){
	int matrix_size = N * N * sizeof(float);
	int vector_size = N * sizeof(float);

	CUDA_HANDLE_ERROR(cudaMalloc((void**)device_matrix, matrix_size));
	CUDA_HANDLE_ERROR(cudaMallocHost((void**)host_matrix, matrix_size));
	CUDA_HANDLE_ERROR(cudaMalloc((void**)device_vector, vector_size));
	CUDA_HANDLE_ERROR(cudaMallocHost((void**)host_vector, vector_size));

	std::mt19937 mt(0);
	std::uniform_real_distribution<float> dist_value(-1.0,1.0);
	for(int i = 0;i < N;i++){
		(*host_vector)[i] = dist_value(mt);
		for(int j = 0;j < N;j++){
			(*host_matrix)[i+N*j] = dist_value(mt);
		}
	}

	CUDA_HANDLE_ERROR(cudaMemcpy(*device_matrix, *host_matrix, matrix_size, cudaMemcpyHostToDevice));
	CUDA_HANDLE_ERROR(cudaMemcpy(*device_vector, *host_vector, vector_size, cudaMemcpyHostToDevice));
	return 0;
}

void showResult(float *host_vector, float *device_vector){
	int vector_size = N * sizeof(float);
	cudaMemcpy( host_vector, device_vector, vector_size, cudaMemcpyDeviceToHost);
}

void freeMatrixAndVector(
		float *host_matrix,
		float *device_matrix,
		float *host_vector,
		float *device_vector){
	CUDA_HANDLE_ERROR(cudaFree( device_matrix ));
	CUDA_HANDLE_ERROR(cudaFree( device_vector ));
	CUDA_HANDLE_ERROR(cudaFreeHost( host_matrix ));
	CUDA_HANDLE_ERROR(cudaFreeHost( host_vector ));
}
int main(){
	//float *device_matrix = NULL,*host_matrix = NULL;
	//float *device_vector = NULL,*host_vector = NULL;
	float *device_matrix ,*host_matrix ;
	float *device_vector ,*host_vector ;

	if( initMatrixAndVector( &host_matrix,&device_matrix,&host_vector,&device_vector ) != 0 ){
		std::cerr<<"error @initMatrixAndVector"<<std::endl;
		return 1;
	}

	std::mt19937 mt(0);
	std::uniform_real_distribution<float> dist_value(-1.0,1.0);
	for(int i = 0;i < N;i++){
		host_vector[i] = dist_value(mt);
		for(int j = 0;j < N;j++){
			host_matrix[i+N*j] = dist_value(mt);
		}
	}

#ifdef USE_CUBLAS
	cublasHandle_t cublas_handle;
	CUBLAS_HANDLE_ERROR(cublasCreate( &cublas_handle ));
	float alpha = 1.0f/N;
	float beta = 0.0f;
#endif

	cudaEvent_t start,stop;
	float elapsed_time;
	CUDA_HANDLE_ERROR(cudaEventCreate( &start ));
	CUDA_HANDLE_ERROR(cudaEventCreate( &stop));
	CUDA_HANDLE_ERROR(cudaEventRecord( start , 0));

	//計算
	for(int calc = 0;calc < CALC;calc++){
#ifdef USE_CUBLAS
		CUBLAS_HANDLE_ERROR(cublasSgemv(
				cublas_handle,
				CUBLAS_OP_N ,
				N,
				N,
				&alpha,
				device_matrix, N,
				device_vector, 1,
				&beta,
				device_vector, 1));
#else
#endif
	}
	//showResult(host_vector,device_vector);
	CUDA_HANDLE_ERROR(cudaEventRecord( stop , 0));
	CUDA_HANDLE_ERROR(cudaEventSynchronize( stop ));
	CUDA_HANDLE_ERROR(cudaEventElapsedTime( &elapsed_time, start, stop));
	std::cout<<"elapsed_time = "<<(elapsed_time/CALC)<<"[ms]"<<std::endl;
#ifdef USE_CUBLAS
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas_handle ));
#endif

	freeMatrixAndVector( host_matrix, device_matrix, host_vector, device_vector );
}
