﻿#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

#define THREAD_PER_BLOCK 256


__device__ void warpReduce(volatile float* cache, unsigned int tid) {
	cache[tid] += cache[tid + 32];
	//__syncthreads();
	cache[tid] += cache[tid + 16];
	//__syncthreads();
	cache[tid] += cache[tid + 8];
	//__syncthreads();
	cache[tid] += cache[tid + 4];
	//__syncthreads();
	cache[tid] += cache[tid + 2];
	//__syncthreads();
	cache[tid] += cache[tid + 1];
	//__syncthreads();
}

__global__ void reduce4(float* d_in, float* d_out) {
	__shared__ float sdata[THREAD_PER_BLOCK];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = d_in[i] + d_in[i + blockDim.x];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

bool check(float* out, float* res, int n) {
	for (int i = 0; i < n; i++) {
		if (out[i] != res[i])
			return false;
	}
	return true;
}

int main() {
	const int N = 32 * 1024 * 1024;
	float* a = (float*)malloc(N * sizeof(float));
	float* d_a;
	cudaMalloc((void**)&d_a, N * sizeof(float));

	int NUM_PER_BLOCK = 2 * THREAD_PER_BLOCK;
	int block_num = N / NUM_PER_BLOCK;
	float* out = (float*)malloc(block_num * sizeof(float));
	float* d_out;
	cudaMalloc((void**)&d_out, block_num * sizeof(float));
	float* res = (float*)malloc(block_num * sizeof(float));

	for (int i = 0; i < N; i++) {
		a[i] = 1;
	}

	for (int i = 0; i < block_num; i++) {
		float cur = 0;
		for (int j = 0; j < NUM_PER_BLOCK; j++) {
			cur += a[i * NUM_PER_BLOCK + j];
		}
		res[i] = cur;
	}

	cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 Grid(block_num, 1);
	dim3 Block(THREAD_PER_BLOCK, 1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float msecTotal = 0;
	int nIter = 1000;
	cudaEventRecord(start);
	for (int i = 0; i < nIter; i++) {
		reduce4 << <Grid, Block >> > (d_a, d_out);

	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);

	if (check(out, res, block_num))printf("the ans is right, time : %f\n", msecTotal);
	else {
		printf("the ans is wrong\n");
		for (int i = 0; i < block_num; i++) {
			printf("%lf ", out[i]);
		}
		printf("\n");
	}

	cudaFree(d_a);
	cudaFree(d_out);
}