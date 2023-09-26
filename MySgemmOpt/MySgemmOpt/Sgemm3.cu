#include "sgemm3.cuh"
#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K,
	const int THREAD_SIZE_Y,
	const int THREAD_SIZE_X
>
__global__ void Sgemm3(float* A, float* B, float* C, const int M, const int N, const int K) {
	//对shared 进行分块,使用float4取数据
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = ty * blockDim.x + tx;
	__shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
	float frag_A[THREAD_SIZE_Y];
	float frag_B[THREAD_SIZE_X];

	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0 };

	const int THREAD_PRE_BLOCK = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_SIZE_Y * THREAD_SIZE_X);
	const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
	const int A_START_ROW = tid / A_THREAD_PER_ROW;
	const int A_START_COL = tid % A_THREAD_PER_ROW * 4;
	const int A_ROW_STRIDE = THREAD_PRE_BLOCK / A_THREAD_PER_ROW;

	const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
	const int B_START_ROW = tid / B_THREAD_PER_ROW;
	const int B_START_COL = tid % B_THREAD_PER_ROW * 4;
	const int B_ROW_STRIDE = THREAD_PRE_BLOCK / B_THREAD_PER_ROW;

	const int ldg_num = BLOCK_SIZE_M * BLOCK_SIZE_K / (4 * THREAD_PRE_BLOCK);
	float ldg_A[4 * ldg_num];

	A = &A[by * BLOCK_SIZE_M * K];
	B = &B[bx * BLOCK_SIZE_N];
#pragma unroll
	for (int i = 0; i < K; i += BLOCK_SIZE_K) {
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_M; j += A_ROW_STRIDE) {
			int ldg_idx = j / A_ROW_STRIDE * 4;
			FETCH_FLOAT4(ldg_A[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(A_START_ROW + j, A_START_COL + i, K)]);
			As[A_START_COL][A_START_ROW + j] = ldg_A[ldg_idx];
			As[A_START_COL + 1][A_START_ROW + j] = ldg_A[ldg_idx + 1];
			As[A_START_COL + 2][A_START_ROW + j] = ldg_A[ldg_idx + 2];
			As[A_START_COL + 3][A_START_ROW + j] = ldg_A[ldg_idx + 3];

		}
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_K; j += B_ROW_STRIDE) {
			FETCH_FLOAT4(Bs[B_START_ROW + j][B_START_COL]) = FETCH_FLOAT4(B[OFFSET(B_START_ROW + j + i, B_START_COL, N)]);
		}

		__syncthreads();
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_K; j++) {
#pragma unroll
			for (int k = 0; k < THREAD_SIZE_Y; k += 4) {
				FETCH_FLOAT4(frag_A[k]) = FETCH_FLOAT4(As[j][ty * THREAD_SIZE_Y + k]);
			}
#pragma unroll
			for (int k = 0; k < THREAD_SIZE_X; k += 4) {
				FETCH_FLOAT4(frag_B[k]) = FETCH_FLOAT4(Bs[j][tx * THREAD_SIZE_X + k]);
			}
#pragma unroll
			for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
				for (int n = 0; n < THREAD_SIZE_X; n++) {
					accum[m][n] += frag_A[m] * frag_B[n];
				}
			}
		}
		__syncthreads();
	}
#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
		for (int n = 0; n < THREAD_SIZE_X; n += 4) {
			FETCH_FLOAT4(C[OFFSET(
				by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + m,
				bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + n,
				N)]) = FETCH_FLOAT4(accum[m][n]);
		}
	}
}
void invokSgemm3() {
	const int m = 2048;
	const int n = 2048;
	const int k = 2048;

	size_t bytes_A = m * k * sizeof(float);
	size_t bytes_B = k * n * sizeof(float);
	size_t bytes_C = m * n * sizeof(float);
	float* A = (float*)malloc(bytes_A);
	float* B = (float*)malloc(bytes_B);
	float* C = (float*)malloc(bytes_C);
	float* C1 = (float*)malloc(bytes_C);

	float* dA, * dB, * dC;
	cudaMalloc((void**)&dA, bytes_A);
	cudaMalloc((void**)&dB, bytes_B);
	cudaMalloc((void**)&dC, bytes_C);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			A[i * k + j] = 1.0;
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			B[i * k + j] = 1.0;
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i * k + j] = -1.0;
		}
	}

	//测量参数配置
	double msecPerMatrixMul[2] = { 0, 0 };
	double gigaFlops[2] = { 0, 0 };
	double flopsPerMatrixMul = 2.0 * m * n * k;

	//核函数参数配置
	const int BLOCK_SIZE_M = 128;
	const int BLOCK_SIZE_N = 128;
	const int BLOCK_SIZE_K = 8;
	const int THREAD_SIZE_Y = 8;
	const int THREAD_SIZE_X = 8;
	//dim3 gridsize(m / 128, n / 128) 不能写死啊
	dim3 gridsize(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
	dim3 blocksize(BLOCK_SIZE_N / THREAD_SIZE_Y, BLOCK_SIZE_M / THREAD_SIZE_X);

	//拷贝数据到device
	cudaMemcpy(dA, A, bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);

	//运行核函数
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float msecTotal = 0;
	int nIter = 1000;
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		Sgemm3<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X> << <gridsize, blocksize >> > (dA, dB, dC, m, n, k);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost);

	msecPerMatrixMul[0] = msecTotal / nIter;
	gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
	printf("My gemm3 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[0],
		msecPerMatrixMul[0],
		flopsPerMatrixMul);

	//运行cublas
	cublasHandle_t blas_handle;
	cublasCreate(&blas_handle);
	float alpha = 1.0;
	float beta = 0;
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
			m, n, k, &alpha,
			dA, k, dB, n, &beta, dC, m
		);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cudaMemcpy(C1, dC, bytes_C, cudaMemcpyDeviceToHost);

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
	printf("ratio= %f%%\n", gigaFlops[0] / gigaFlops[1] * 100);

	//释放内存
	free(A);
	free(B);
	free(C);
	free(C1);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}