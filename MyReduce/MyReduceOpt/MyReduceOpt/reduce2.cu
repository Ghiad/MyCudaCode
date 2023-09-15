#include "reduce2.cuh"

__global__ void reduce2(float* d_in, float* d_out)
{
    __shared__ float sdata[Thread_per_block];

    int tid = threadIdx.x;
    sdata[tid] = d_in[blockDim.x * blockIdx.x + tid];

    __syncthreads();

    //解决shared memory bank conflict
    for (int s = blockDim.x/2; s >0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}