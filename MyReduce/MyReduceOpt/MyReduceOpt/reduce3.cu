#include "reduce3.cuh"

__global__ void reduce3(float* d_in, float* d_out)
{
    __shared__ float sdata[Thread_per_block];

    int tid = threadIdx.x;
    int id = 2 * blockDim.x * blockIdx.x + tid;
    //解决idle thread
    sdata[tid] = d_in[id] + d_in[id+blockDim.x];
    __syncthreads();

    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}