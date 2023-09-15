#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>
using namespace std;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int BLOCK_SIZE_K,
    const int THREAD_SIZE_Y,
    const int THREAD_SIZE_X
>
__global__ void Sgemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C,const int M,const int N,const int K){
    //参数准备
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    //针对C分配线程
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;
    
    //int tid = tx + ty * blockDim.x;和下面效果应该相等
    int tid = tx + ty * THREAD_X_PER_BLOCK;

    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int A_TILE_ROW = tid / A_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;

    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    const int B_TILE_ROW = tid / B_TILE_THREAD_PER_ROW;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
    // 每个线程一次取4个float,需要取多少次
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (4 * THREAD_NUM_PER_BLOCK);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (4 * THREAD_NUM_PER_BLOCK);
    //双倍缓存
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    float frag_A[2][THREAD_SIZE_Y];
    float frag_B[2][THREAD_SIZE_X];

    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    //方便后续global memory的读取，我只关注我要读取的global中在这次大迭代中会取到的数据
    A = &A[by * BLOCK_SIZE_M * K];
    B = &B[bx * BLOCK_SIZE_N];

    //预取数据
    //预取A --> As,索引计算
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        //我们这只是取第一次的数据所以列方面不需要加上bx*bk
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(A_TILE_ROW+i, A_TILE_COL, K)]);

        As[0][A_TILE_COL][A_TILE_ROW] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL +1][A_TILE_ROW] = ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL +2][A_TILE_ROW] = ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL +3][A_TILE_ROW] = ldg_a_reg[ldg_index+3];
    }
    //预取B --> Bs,为啥b不用通过寄存器
    for (int i = 0; i < BLOCK_SIZE_M; i += B_TILE_ROW_STRIDE) {       
        FETCH_FLOAT4(Bs[0][B_TILE_ROW][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(B_TILE_ROW+i,B_TILE_COL,N)]);
    }
    __syncthreads();
    //预取As-->frag_A,考虑扩展性不要写8，而是thread_size_y
    for (int i = 0; i < THREAD_SIZE_Y; i+=4) {
        //FETCH_FLOAT4(frag_A[0][i]) = FETCH_FLOAT4(As[0][0][ty * 8 + i]);
        FETCH_FLOAT4(frag_A[0][i]) = FETCH_FLOAT4(As[0][0][ty * THREAD_SIZE_Y + i]);
    }
    //预取Bs-->frag_B
    for (int i = 0; i < THREAD_SIZE_X; i+=4) {
        FETCH_FLOAT4(frag_B[0][i]) = FETCH_FLOAT4(Bs[0][0][tx * THREAD_SIZE_X + i]);
    }

    int write_stage_idx = 1;
    int load_stage_idx = write_stage_idx ^ 1;
    //开始大迭代
    for (int i = 0; i < K;) {
        i += BLOCK_SIZE_K;
        //如果还有下一次循环，那么就预取数据，从global-->ldg
        if (i < K) {
            for (int j = 0; j < BLOCK_SIZE_M; j += A_TILE_ROW_STRIDE) {
                int ldg_index = j / A_TILE_ROW_STRIDE * 4;
                //大迭代还是在一个block上的概念而不是Block间
                //FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(A_TILE_ROW + j, A_TILE_COL + bx * BLOCK_SIZE_K, K)]);
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(A_TILE_ROW + j, A_TILE_COL + i, K)]);
            }
            for (int j = 0; j < BLOCK_SIZE_K; j += B_TILE_ROW_STRIDE) {
                int ldg_index = j / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(B_TILE_ROW + j + i, B_TILE_COL, N)]);
            }
        }
        //开始小循环
        for (int j = 0; j < BLOCK_SIZE_K - 1; j++) {
            for (int k = 0; k < THREAD_SIZE_Y; k+=4) {
                FETCH_FLOAT4(frag_A[(j + 1) % 2][k]) = FETCH_FLOAT4(As[load_stage_idx][j + 1][ty * THREAD_SIZE_Y + k]);
            }
            for (int k = 0; k < THREAD_SIZE_Y; k += 4) {
                FETCH_FLOAT4(frag_B[(j + 1) % 2][k]) = FETCH_FLOAT4(Bs[load_stage_idx][j + 1][tx * THREAD_SIZE_X + k]);
            }
            for (int m = 0; m < THREAD_SIZE_Y; m++) {
                for (int n = 0; n < THREAD_SIZE_X; n++) {
                    //如果上次预取的数据还没取完,会怎么样
                    accum[m][n] += frag_A[j % 2][m] * frag_B[j % 2][n];
                }
            }
        }

        //如果还有下一次循环，预取数据从ldg-->shared
        if (i < K) {
            for (int j = 0; j < BLOCK_SIZE_M; j += A_TILE_ROW_STRIDE) {
                int idx = j / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW + j] = ldg_a_reg[idx];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW + j] = ldg_a_reg[idx +1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW + j] = ldg_a_reg[idx +2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW + j] = ldg_a_reg[idx +3];
            }
            for (int j = 0; j < BLOCK_SIZE_K; j += B_TILE_ROW_STRIDE) {
                int idx = j / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW + j][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[idx]);
            }
            //对shared memory需要同步
            __syncthreads();
            write_stage_idx ^= 1;
            load_stage_idx = write_stage_idx ^ 1;
        }
        
        //计算最后一次小循环,BLOCK_SIZE_K一般是偶数,所以上面的循环必定会在为frag_A[1]存数据时停止，
        //同时要为下一次大循环预取数据
        for (int k = 0; k < THREAD_SIZE_Y; k += 4) {
            FETCH_FLOAT4(frag_A[0][k]) = FETCH_FLOAT4(As[load_stage_idx][0][ty * THREAD_SIZE_Y + k]);
        }
        for (int k = 0; k < THREAD_SIZE_Y; k += 4) {
            FETCH_FLOAT4(frag_B[0][k]) = FETCH_FLOAT4(Bs[load_stage_idx][0][tx * THREAD_SIZE_X + k]);
        }
        for (int m = 0; m < THREAD_SIZE_Y; m++) {
            for (int n = 0; n < THREAD_SIZE_X; n++) {
                //如果上次预取的数据还没取完,会怎么样
                accum[m][n] += frag_A[1][m] * frag_B[1][n];
            }
        }

    }
    //这里为啥不需要__syncthreads();因为accum是每个线程独有的，不需要一致性
    C = &C[OFFSET(by * BLOCK_SIZE_M, bx * BLOCK_SIZE_K, K)];
    for (int m = 0; m < THREAD_SIZE_Y; m+=4) {
        for (int n = 0; n < THREAD_SIZE_X; n+=4) {
            FETCH_FLOAT4(C[OFFSET(ty * THREAD_SIZE_Y + m, tx * THREAD_SIZE_X + n,N)]) = FETCH_FLOAT4(accum[m][n]);
             
        }
    }
   

}


int main() {
    const int m = 2048;
    const int n = 2048;
    const int k = 2048;
    //数据空间分配
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);
    float* A = (float*)malloc(bytes_A);
    float* B = (float*)malloc(bytes_B);
    float* C = (float*)malloc(bytes_C );
    float* C1 = (float*)malloc(bytes_C);
    float* dA, * dB, * dC;
    checkCudaErrors(cudaMalloc((void**)&dA, bytes_A));
    checkCudaErrors(cudaMalloc((void**)&dB, bytes_B));
    checkCudaErrors(cudaMalloc((void**)&dC, bytes_C ));
    
    //数据初始化
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[i*k+j] = 1.0;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B[i * k + j] = 1.0;
        }
    }
    //测量参数配置
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = { 0, 0 };
    double flopsPerMatrixMul = 2.0 * m * n * k;

    //核函数参数配置
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_SIZE_X = 8;
    dim3 gridsize(m / 128, n / 128);
    dim3 blocksize(BLOCK_SIZE_M / THREAD_SIZE_Y, BLOCK_SIZE_N / THREAD_SIZE_X);

    //拷贝数据到device
    checkCudaErrors(cudaMemcpy(dA, A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dB, B, bytes_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dC, C, bytes_C , cudaMemcpyHostToDevice));

    //运行核函数
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1;
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        Sgemm <BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X> << <gridsize, blocksize >> > (dA, dB, dC, m, n, k);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    //运行cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
            m, n, k, &alpha,
            dA, k, dB, n, &beta, dC, n
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaMemcpy(C1, dC, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);
    cublasDestroy(blas_handle);

    //检测运行结果的正确性
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        int row = i / n;
        int col = i % n;
        double abs_err = fabs(C[i] - C1[col * m + row]);
        double dot_length = m;
        double abs_val = fabs(C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                i, C[i], C1[col * m + row], eps);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    //释放内存
    free(A);
    free(B);
    free(C);
    free(C1);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}