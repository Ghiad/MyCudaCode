#include "reduce4.cuh"

__device__ void warpReduce(volatile float* sdata,int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4(float* d_in, float* d_out)
{
    __shared__ float sdata[Thread_per_block];

    int tid = threadIdx.x;
    int id = 2 * blockDim.x * blockIdx.x + tid;
    sdata[tid] = d_in[id] + d_in[id + blockDim.x];
    __syncthreads();

    //展开最后一维
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid<32) warpReduce(sdata,tid);

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}