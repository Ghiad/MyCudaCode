#include "Sgemm2.cuh"

//利用shared memory解决global memory重复读取的问题，每个线程解决一个元素的计算
template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K
>
__global__ void Sgemm2(float* A, float* B, float* C, const int M, const int N, const int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = by * blockDim.y + ty;
	int y = bx * blockDim.x + tx;
	//对A进行转置
	__shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
	float accum = 0.f;
	for (int i = 0; i < K; i += BLOCK_SIZE_K) {
		//一个线程负责读一个数据,相邻的线程读取相邻的元素,有利于合并访存
		As[tx][ty] = A[x * K + tx + i];//按行读按列存
		Bs[ty][tx] = B[(ty + i) * K + y];//按行读按行存
		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE_M; j++) {
			accum += As[j][ty] * Bs[j][tx];
		}
		__syncthreads;
	}
	C[x * K + y] = accum;
}

void invokSgemm2() {
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
	const int BLOCK_SIZE_M = 16;
	const int BLOCK_SIZE_N = 16;
	const int BLOCK_SIZE_K = 16;
	const int THREAD_SIZE_Y = 1;
	const int THREAD_SIZE_X = 1;
	//dim3 gridsize(m / 128, n / 128) 不能写死啊
	dim3 gridsize(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
	dim3 blocksize(BLOCK_SIZE_M / THREAD_SIZE_Y, BLOCK_SIZE_K / THREAD_SIZE_X);

	//拷贝数据到device
	cudaMemcpy(dA, A, bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);

	//运行核函数
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float msecTotal = 0;
	int nIter = 100;
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		Sgemm2<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K> << <gridsize, blocksize >> > (dA, dB, dC, m, n, k);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost);

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
	printf("ratio= %f%%\n", gigaFlops[0] / gigaFlops[1]*100);

	//释放内存
	free(A);
	free(B);
	free(C);
	free(C1);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}