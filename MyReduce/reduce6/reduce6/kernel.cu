#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>
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
__device__ void warpReduce(volatile float* cache, int tid) {
    if (blockSize >= 64)cache[tid] += cache[tid + 32];
    if (blockSize >= 32)cache[tid] += cache[tid + 16];
    if (blockSize >= 16)cache[tid] += cache[tid + 8];
    if (blockSize >= 8)cache[tid] += cache[tid + 4];
    if (blockSize >= 4)cache[tid] += cache[tid + 2];
    if (blockSize >= 2)cache[tid] += cache[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce6(float* d_in, float* d_out, int n)
{
    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int id = (2 * blockSize) * blockIdx.x + tid;
    unsigned int gridSize = 2 * blockSize * gridDim.x;
    //sdata[tid] = 0.0;

    float tmp_sdata = 0.0f; 
    while (id < n) { 
        tmp_sdata += d_in[id] + d_in[id + blockSize];
        id += gridSize;
    }
    sdata[tid] = tmp_sdata;
    __syncthreads();

    //完全展开
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
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
        reduce6<Thread_per_block> << <gridsize, blocksize >> > (d_in, d_out,N);
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