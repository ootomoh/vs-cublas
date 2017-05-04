#include <stdio.h>
const int THREADS = 1<<8;
// s  : single
// ge : general
// mv : matrix vector
__global__ void kernelSgemv0(int n,int s,const float alpha,const float *A,const float *x,float beta,float *y){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int loop_begin = tid * s;
	int loop_end = loop_begin + s;
	float dot;
	float *res_y;
	for(int j = loop_begin;j < loop_end;j++){
		dot = 0.0f;
		for(int i = 0;i < n;i++){
			dot += alpha * A[i*n+j] * x[i];
		}
		res_y = y + j;
		*res_y = beta * *res_y + dot;
	}
}

__global__ void kernelSgemv1(int n,const float *A,const float *x,float *y){
	__shared__ float cache[THREADS];
	int j = blockIdx.x;
	int i = threadIdx.x;
	cache[i] = 0.0f;
	while(i < n){
		cache[threadIdx.x] += A[j+i*n]*x[i];
		i+=THREADS;
	}
	__syncthreads();
	int mod = 1;
	while(mod<THREADS){
		mod<<=1;
		if(threadIdx.x%mod==0){
			cache[threadIdx.x] += cache[threadIdx.x+mod/2];
		}
		__syncthreads();
	};
	y[j] = cache[0];
}

void mySgemv(int m,int n,const float *alpha,const float *A,const float *x,float *beta,float *y){
	int threads = (n > THREADS ? THREADS : n);
	kernelSgemv0<<<n/threads,threads>>>(n,1,*alpha,A,x,*beta,y);
	//kernelSgemv1<<<n,THREADS>>>(n,A,x,y);
}
