#include "reduce6.cuh"
#include<iostream>
#include <malloc.h>

using namespace std;


bool check(float* out, float* res, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != res[i])
            return false;
    }
    return true;
}

int main()
{
    int block_num = N / Num_per_block;
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
    cudaMalloc((void**)&d_out,block_num*sizeof(float));
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    dim3 gridsize(block_num);
    dim3 blocksize(Thread_per_block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        reduce6<Thread_per_block><<<gridsize, blocksize >>> (d_in, d_out,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float   elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float* out = (float*)malloc(block_num * sizeof(float));
    cudaMemcpy(out, d_out, N / Num_per_block * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  
    if (check(out, res, block_num))printf("the ans is right\n");
    else {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++) {
            printf("%lf ", out[i]);
        }
        printf("\n");
    }
    cout << "Time is " << elapsedTime <<" ms " << endl;
    cudaFree(d_in);
    cudaFree(d_out);
    free(in);


    return 0;
}

