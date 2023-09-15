#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>
#define WARP_SIZE 32
using namespace std;

const long int N = 32 * 1024 * 1024;

const int Thread_per_block = 256;

bool check(float* out, float* res, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != res[i])
            return false;
    }
    return true;
}
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce7(float* d_in, float* d_out, int n)
{
    float sum = 0;

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    #pragma unroll
    for (int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if (laneId == 0)warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed    
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize / WARP_SIZE>(sum);
    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sum;
}

int main()
{
    const int block_num = 2048;
    const int Num_per_block = N / block_num;
    const int Num_per_thread = Num_per_block / Thread_per_block;

    long int size = N * sizeof(float);
    float* in = (float*)malloc(size);
    float* res = (float*)malloc(block_num * sizeof(float));
    float* d_in;
    float* d_out;

    for (long int i = 0; i < N; i++) {
        in[i] = 1.0;
    }
    for (int i = 0; i < block_num; i++) {
        float cur = 0;
        for (int j = 0; j < Num_per_block; j++) {
            cur += in[i * Num_per_block + j];
        }
        res[i] = cur;
    }

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, block_num * sizeof(float));
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    dim3 gridsize(block_num);
    dim3 blocksize(Thread_per_block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        reduce7<Thread_per_block,Num_per_thread> << <gridsize, blocksize >> > (d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float* out = (float*)malloc(block_num * sizeof(float));
    cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (check(out, res, block_num))printf("the ans is right\n");
    else {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++) {
            printf("%lf ", out[i]);
        }
        printf("\n");
    }
    cout << "Time is " << elapsedTime << " ms " << endl;
    cudaFree(d_in);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(in);


    return 0;
}