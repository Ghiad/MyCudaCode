#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//#include "reduce.cuh"
//#include "GEMV.cuh"
//#include "test.cuh"
#include "GEMM.cuh"
//#include"Sgemm4.cuh"
using namespace std;
#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K,
	const int THREAD_SIZE_Y,
	const int THREAD_SIZE_X
>
__global__ void SGEMM(float* A, float* B, float* C, int M, int N, int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;


	int tid = ty * blockDim.x + tx;

	__shared__ float s_a[BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0 };
	float r_a[THREAD_SIZE_Y];
	float r_b[THREAD_SIZE_X];
	const int threads_per_block = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_SIZE_Y * THREAD_SIZE_X);
	const int A_threads_per_row = BLOCK_SIZE_K / 4;
	const int B_threads_per_row = BLOCK_SIZE_N / 4;
	const int A_stride = threads_per_block / A_threads_per_row;
	const int B_stride = threads_per_block / B_threads_per_row;
	const int A_start_row = tid / A_threads_per_row;
	const int A_start_col = tid % A_threads_per_row * 4;
	const int B_start_row = tid / B_threads_per_row;
	const int B_start_col = tid % B_threads_per_row * 4;
	const int ldg_num = (BLOCK_SIZE_M * BLOCK_SIZE_N) / 4 * threads_per_block;
	float ldg_A[4 * ldg_num];//感觉4就够了


	A = &A[by * BLOCK_SIZE_M * K];
	B = &B[bx * BLOCK_SIZE_N];
#pragma unroll
	for (int i = 0; i < K; i += BLOCK_SIZE_K) {
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_M; j += A_stride) {
			int ldg_index = j / A_stride * 4;
			FETCH_FLOAT4(ldg_A[ldg_index]) = FETCH_FLOAT4(A[i + (A_start_row + j) * K + A_start_col]);
			s_a[A_start_col][A_start_row + j] = ldg_A[ldg_index];
			s_a[A_start_col + 1][A_start_row + j] = ldg_A[ldg_index + 1];
			s_a[A_start_col + 2][A_start_row + j] = ldg_A[ldg_index + 2];
			s_a[A_start_col + 3][A_start_row + j] = ldg_A[ldg_index + 3];
		}
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_K; j += B_stride) {
			FETCH_FLOAT4(s_b[B_start_row + j][B_start_col]) = FETCH_FLOAT4(B[(B_start_row + j + i) * N + B_start_col]);
		}
		__syncthreads();

#pragma unroll
		for (int m = 0; m < BLOCK_SIZE_K; m++) {
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_Y; j += 4) {
				//r_a[j] = s_a[m][j + ty * THREAD_SIZE_Y];
				FETCH_FLOAT4(r_a[j]) = FETCH_FLOAT4(s_a[m][j + ty * THREAD_SIZE_Y]);
			}
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_X; j += 4) {
				//r_b[j] = s_b[m][j + tx * THREAD_SIZE_X];
				FETCH_FLOAT4(r_b[j]) = FETCH_FLOAT4(s_b[m][j + tx * THREAD_SIZE_X]);
			}
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_Y; j++) {
#pragma unroll
				for (int l = 0; l < THREAD_SIZE_X; l++) {
					accum[j][l] += r_a[j] * r_b[l];
				}
			}
		}

		__syncthreads();
	}
#pragma unroll
	for (int j = 0; j < THREAD_SIZE_Y; j++) {
#pragma unroll
		for (int l = 0; l < THREAD_SIZE_X; l += 4) {
			//printf("%d %d %d %f\n", bx, by, tid, accum[j][l]);
			FETCH_FLOAT4(C[(ty * THREAD_SIZE_Y + by * BLOCK_SIZE_M + j) * N + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + l])
				= FETCH_FLOAT4(accum[j][l]);
			//C[(ty * THREAD_SIZE_Y + by * BLOCK_SIZE_M + j) * N + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + l] = accum[j][l];
		}
	}
}


int sgemm() {
	const int m = 1024;
	const int n = 1024;
	const int k = 1024;
	//数据空间分配
	size_t bytes_A = m * k * sizeof(float);
	size_t bytes_B = k * n * sizeof(float);
	size_t bytes_C = m * n * sizeof(float);
	float* A = (float*)malloc(bytes_A);
	float* B = (float*)malloc(bytes_B);
	float* C = (float*)malloc(bytes_C);

	float* C1 = (float*)malloc(bytes_C);
	float* dA, * dB, * dC;
	cudaMalloc((void**)&dA, bytes_A);
	cudaMalloc((void**)&dB, bytes_B);
	cudaMalloc((void**)&dC, bytes_C);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			A[i * k + j] = 1.0;
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			B[i * k + j] = 1.0;
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i * k + j] = 0.0;
		}
	}
	//测量参数配置
	double msecPerMatrixMul[2] = { 0, 0 };
	double gigaFlops[2] = { 0, 0 };
	double flopsPerMatrixMul = 2.0 * m * n * k;

	//核函数参数配置
	const int BLOCK_SIZE_M = 128;
	const int BLOCK_SIZE_N = 128;
	const int BLOCK_SIZE_K = 8;
	const int THREAD_SIZE_Y = 8;
	const int THREAD_SIZE_X = 8;
	//dim3 gridsize(m / 128, n / 128) 不能写死啊
	dim3 gridsize(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
	dim3 blocksize(BLOCK_SIZE_N / THREAD_SIZE_Y, BLOCK_SIZE_M / THREAD_SIZE_X);

	//拷贝数据到device
	cudaMemcpy(dA, A, bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);

	//test << <1, dim3(5, 2) >> > (dC);
	//cudaDeviceSynchronize();

	//运行核函数
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float msecTotal = 0;
	int nIter = 1000;
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		SGEMM<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X> << <gridsize, blocksize >> > (dA, dB, dC, m, n, k);
		/*cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
		}*/
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost);

	msecPerMatrixMul[0] = msecTotal / nIter;
	gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
	printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[0],
		msecPerMatrixMul[0],
		flopsPerMatrixMul);

	//运行cublas
	cublasHandle_t blas_handle;
	cublasCreate(&blas_handle);
	float alpha = 1.0;
	float beta = 0;
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
			m, n, k, &alpha,
			dA, k, dB, n, &beta, dC, m
		);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cudaMemcpy(C1, dC, bytes_C, cudaMemcpyDeviceToHost);

	msecPerMatrixMul[1] = msecTotal / nIter;
	gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
	printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[1],
		msecPerMatrixMul[1],
		flopsPerMatrixMul);
	cublasDestroy(blas_handle);

	//检测运行结果的正确性
	double eps = 1.e-6;  // machine zero
	bool correct = true;
	for (int i = 0; i < m * n; i++) {
		int row = i / n;
		int col = i % n;
		double abs_err = fabs(C[i] - C1[col * m + row]);
		double dot_length = m;
		double abs_val = fabs(C[i]);
		double rel_err = abs_err / abs_val / dot_length;
		if (rel_err > eps) {
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, C[i], C1[col * m + row], eps);
			correct = false;
			break;
		}
	}
	printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
	printf("ratio= %f%%\n", gigaFlops[0] / gigaFlops[1] * 100);

	//释放内存
	free(A);
	free(B);
	free(C);
	free(C1);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	return 0;
}
int main() {
	//invokSgemm4();
	//cudaDeviceSynchronize();
	gemm();
	return 0;
}