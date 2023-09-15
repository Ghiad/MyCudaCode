#include "reduce0.cuh"

__global__ void reduce0(float* d_in, float* d_out)
{
    __shared__ float sdata[Thread_per_block];
    
    int tid = threadIdx.x;
    sdata[tid] = d_in[blockDim.x * blockIdx.x + tid];
    
    __syncthreads();


    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}