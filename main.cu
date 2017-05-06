#include <iostream>
#include <random>
#include <boost/program_options.hpp>

// matrix : 2^N x 2^N
// vector : 2^N
const int N = 12;
// how many calculations
const int CALC = 1<<12;

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
void mySgemv(int m,int n,const float *alpha,const float *A,const float* x,float *beta,float *y);

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

class SgemvTest{
	int matrix_size,vector_size,n;
	float *host_matrix;
	float *device_matrix;
	float *host_vector_0;
	float *device_vector_0;
	float *host_vector_1;
	float *device_vector_1;
	float alpha;
	float beta;
	cublasHandle_t cublas_handle;
public:
	SgemvTest(int n):n(n),matrix_size(n*n*sizeof(float)),vector_size(n*sizeof(float)),alpha(1.0f/n),beta(0){
		CUDA_HANDLE_ERROR(cudaMalloc((void**)&device_matrix, matrix_size));
		CUDA_HANDLE_ERROR(cudaMallocHost((void**)&host_matrix, matrix_size));
		CUDA_HANDLE_ERROR(cudaMalloc((void**)&device_vector_0, vector_size));
		CUDA_HANDLE_ERROR(cudaMallocHost((void**)&host_vector_0, vector_size));
		CUDA_HANDLE_ERROR(cudaMalloc((void**)&device_vector_1, vector_size));
		CUDA_HANDLE_ERROR(cudaMallocHost((void**)&host_vector_1, vector_size));
	CUBLAS_HANDLE_ERROR(cublasCreate( &cublas_handle ));
	}
	~SgemvTest(){
		/*CUDA_HANDLE_ERROR( cudaMemcpy( host_vector_1, device_vector_1 , vector_size,cudaMemcpyDeviceToHost) );
		for(int i = 0;i  <n;i++){
			std::cout<<host_vector_1[i]<<",";
		}
		std::cout<<std::endl;*/
		CUDA_HANDLE_ERROR(cudaFree( device_matrix ));
		CUDA_HANDLE_ERROR(cudaFree( device_vector_0 ));
		CUDA_HANDLE_ERROR(cudaFree( device_vector_1 ));
		CUDA_HANDLE_ERROR(cudaFreeHost( host_matrix ));
		CUDA_HANDLE_ERROR(cudaFreeHost( host_vector_0 ));
		CUDA_HANDLE_ERROR(cudaFreeHost( host_vector_1 ));
		CUBLAS_HANDLE_ERROR(cublasDestroy( cublas_handle ));
	}

	void init(){
		std::mt19937 mt(0);
		std::uniform_real_distribution<float> dist_value(-1.0,1.0);
		for(int i = 0;i < n;i++){
			host_vector_0[i] = dist_value(mt);
			for(int j = 0;j < n;j++){
				host_matrix[i+n*j] = dist_value(mt);
			}
		}
		CUDA_HANDLE_ERROR(cudaMemcpy(device_matrix, host_matrix, matrix_size, cudaMemcpyHostToDevice));
		CUDA_HANDLE_ERROR(cudaMemcpy(device_vector_0, host_vector_0, vector_size, cudaMemcpyHostToDevice));
		CUDA_HANDLE_ERROR(cudaMemset(device_vector_1, 0, vector_size));
	}

	void cublasSgemvTest(int calc){
		cudaEvent_t start,stop;
		float elapsed_time;
		CUDA_HANDLE_ERROR(cudaEventCreate( &start ));
		CUDA_HANDLE_ERROR(cudaEventCreate( &stop));
		CUDA_HANDLE_ERROR(cudaEventRecord( start , 0));
		for(int c= 0;c< calc;c++){
			CUBLAS_HANDLE_ERROR(cublasSgemv(
						cublas_handle,
						CUBLAS_OP_N ,
						n,
						n,
						&alpha,
						device_matrix, n,
						device_vector_0, 1,
						&beta,
						device_vector_1, 1));
		}
		CUDA_HANDLE_ERROR(cudaEventRecord( stop , 0));
		CUDA_HANDLE_ERROR(cudaEventSynchronize( stop ));
		CUDA_HANDLE_ERROR(cudaEventElapsedTime( &elapsed_time, start, stop));
		std::cout<<"elapsed_time = "<<(elapsed_time/calc)<<"[ms]"<<std::endl;
	}
	void mySgemvTest(int calc){
		cudaEvent_t start,stop;
		float elapsed_time;
		CUDA_HANDLE_ERROR(cudaEventCreate( &start ));
		CUDA_HANDLE_ERROR(cudaEventCreate( &stop));
		CUDA_HANDLE_ERROR(cudaEventRecord( start , 0));
		for(int c= 0;c< calc;c++){
			mySgemv(n,n,&alpha,device_matrix,device_vector_0,&beta,device_vector_1);
		}
		CUDA_HANDLE_ERROR(cudaEventRecord( stop , 0));
		CUDA_HANDLE_ERROR(cudaEventSynchronize( stop ));
		CUDA_HANDLE_ERROR(cudaEventElapsedTime( &elapsed_time, start, stop));
		std::cout<<"elapsed_time = "<<(elapsed_time/calc)<<"[ms]"<<std::endl;
	}
};

int main(int argc,char** argv){
	boost::program_options::options_description opt("Options");
	opt.add_options()
		("help,h",			"show helps")
		("cublas,c",		"cublasSgemv test mode")
		("my,m",			"my sgemv method test mode")
		("number,n",boost::program_options::value<int>()->default_value(N),"matrix size (R^{2^n x 2^n}) (n <= 14)")
		("times,t",boost::program_options::value<int>()->default_value(CALC),"calculation count ");
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc,argv,opt),vm);
	boost::program_options::notify(vm);

	if(vm.count("help") != 0){
		std::cout<<opt<<std::endl;
	}
	int n = vm["number"].as<int>();
	int calc = vm["times"].as<int>();
	if(n < 0 || 14 < n){
		std::cerr<<"n : invalid range"<<std::endl;
		return 1;
	}
	std::cout<<"matrix size : "<<(1<<n)<<" x "<<(1<<n)<<std::endl;
	std::cout<<"calculation count : "<<calc<<std::endl;
	if(vm.count("cublas")){
		SgemvTest sgemTest(1<<n);
		sgemTest.init();
		std::cout<<"-- cublasSgemvTest --"<<std::endl;
		sgemTest.cublasSgemvTest(calc);
	}
	if(vm.count("my")){
		SgemvTest sgemTest(1<<n);
		sgemTest.init();
		std::cout<<"-- mySgemvTest --"<<std::endl;
		sgemTest.mySgemvTest(calc);
	}
}
