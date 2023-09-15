#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>


const long int N = 4 * 1024 * 1024;
const int Num_per_block = 512;
const int Thread_per_block = 256;

template <unsigned int blockSize>
__device__ void warpReduce5(volatile float* cache, int tid) {
    if (blockSize >= 64)cache[tid] += cache[tid + 32];
    if (blockSize >= 32)cache[tid] += cache[tid + 16];
    if (blockSize >= 16)cache[tid] += cache[tid + 8];
    if (blockSize >= 8)cache[tid] += cache[tid + 4];
    if (blockSize >= 4)cache[tid] += cache[tid + 2];
    if (blockSize >= 2)cache[tid] += cache[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce5(float* d_in, float* d_out)
{
    __shared__ float sdata[Thread_per_block];

    int tid = threadIdx.x;
    int id = 2 * blockSize * blockIdx.x + tid;
    sdata[tid] = d_in[id] + d_in[id + blockSize];
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
    if (tid < 32) warpReduce5<blockSize>(sdata, tid);
    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}