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
void mySgemv(int m,int n,const float *alpha,const float *A,const float *x,float *beta,float *y){
	int threads = (n > THREADS ? THREADS : n);
	kernelSgemv0<<<n/threads,threads>>>(n,1,*alpha,A,x,*beta,y);
}
