#include "sgemm3.cuh"
__global__ void Sgemm3(float* A, float* B, float* C, const int M, const int N, const int K) {
	//对shared 进行分块
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;


}