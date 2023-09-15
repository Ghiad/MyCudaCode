#include "reduce1.cuh"

__global__ void reduce1(float* d_in, float* d_out)
{
    __shared__ float sdata[Thread_per_block];

    int tid = threadIdx.x;
    sdata[tid] = d_in[blockDim.x * blockIdx.x + tid];

    __syncthreads();

    //解决线程束分化
    for (int s = 1; s < blockDim.x; s *= 2) {
        int id = tid * 2 * s;
        if (id < blockDim.x) {
            sdata[id] += sdata[id + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];
}