#include"GemmItr0.cuh"
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> ((&pointer))[0])

template<unsigned int BLOCK_SIZE_M,
	unsigned int BLOCK_SIZE_N,
	unsigned int BLOCK_SIZE_K,
	unsigned int THREAD_SIZE_X,
	unsigned int THREAD_SIZE_Y>
__global__ void Gemm(float* A, float* B, float* C, int m, int n, int k) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = tx + ty * blockDim.x;

	__shared__ float shareA[BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float shareB[BLOCK_SIZE_K][BLOCK_SIZE_N];
	float fragA[THREAD_SIZE_Y];
	float fragB[THREAD_SIZE_X];
	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0f };

	const int threadPerBlock = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_SIZE_X * THREAD_SIZE_Y);
	int aThreadPerRow = BLOCK_SIZE_K / 4;
	int aStartRow = tid / aThreadPerRow;
	int aStartCol = (tid & (aThreadPerRow - 1)) * 4;
	int aStride = threadPerBlock / aThreadPerRow;
	const int aLdgNum = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (4 * threadPerBlock);
	float ldgA[4 * aLdgNum];

	int bThreadPerRow = BLOCK_SIZE_N / 4;
	int bStartRow = tid / bThreadPerRow;
	int bStartCol = (tid & (bThreadPerRow - 1)) * 4;
	int bStride = threadPerBlock / bThreadPerRow;

	A = &A[by * k * blockDim.y];
	B = &B[bx * blockDim.x];
	for (int i = 0; i < k; i += BLOCK_SIZE_K) {
		for (int j = 0; j < BLOCK_SIZE_M; j += aStride) {
			int ldgIndex = (j / aStride) * 4;
			FETCH_FLOAT4(ldgA[ldgIndex]) = FETCH_FLOAT4(A[k * (aStartRow + j) + i + aStartCol]);
			shareA[aStartCol][aStartRow] = ldgA[ldgIndex];
			shareA[aStartCol + 1][aStartRow + j] = ldgA[ldgIndex + 1];
			shareA[aStartCol + 2][aStartRow + j] = ldgA[ldgIndex + 2];
			shareA[aStartCol + 3][aStartRow + j] = ldgA[ldgIndex + 3];
		}

		for (int j = 0; j < BLOCK_SIZE_K; j += bStride) {
			FETCH_FLOAT4(shareB[bStartRow + j][bStartCol]) = FETCH_FLOAT4(B[n * (bStartRow + j + i) + bStartCol]);
		}
		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE_K; j++) {
			for (int k = 0; k < THREAD_SIZE_Y; k += 4) {
				FETCH_FLOAT4(fragA[k]) = FETCH_FLOAT4(shareA[j][k + ty * THREAD_SIZE_Y]);
			}
			for (int k = 0; k < THREAD_SIZE_X; k += 4) {
				FETCH_FLOAT4(fragB[k]) = FETCH_FLOAT4(shareB[j][tx * THREAD_SIZE_X + k]);
			}

			for (int m = 0; m < THREAD_SIZE_Y; m++) {
				for (int n = 0; n < THREAD_SIZE_X; n++) {
					accum[m][n] += fragA[m] * fragB[n];
				}
			}
		}
		__syncthreads();
	}

	for (int m = 0; m < THREAD_SIZE_Y; m++) {
		for (int n = 0; n < THREAD_SIZE_X; n += 4) {
			FETCH_FLOAT4(C[n * (by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + m) + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + n])
				= FETCH_FLOAT4(accum[m][n]);
		}
	}
}

void lanuchGemm() {
	const int m = 2048;
	const int n = 2048;
	const int k = 2048;

	unsigned int sizeA = sizeof(float) * m * k;
	unsigned int sizeB = sizeof(float) * n * k;
	unsigned int sizeC = sizeof(float) * m * n;

	float* A = (float*)malloc(sizeA);
	float* B = (float*)malloc(sizeB);
	float* C = (float*)malloc(sizeC);

	float* dA;
	float* dB;
	float* dC;

	cudaMalloc((void**)dA, sizeA);
	cudaMalloc((void**)dB, sizeB);
	cudaMalloc((void**)dC, sizeC);

	cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, sizeC, cudaMemcpyHostToDevice);

	const int BLOCK_SIZE_M = 128;
	const int BLOCK_SIZE_N = 128;
	const int BLOCK_SIZE_K = 8;
	const int THREAD_SIZE_X = 8;
	const int THREAD_SIZE_Y = 8;

	dim3 blockSize(BLOCK_SIZE_M / THREAD_SIZE_Y, BLOCK_SIZE_N / THREAD_SIZE_X);
	dim3 gridSize(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);

	Gemm<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y> << <gridSize, blockSize >> > (dA, dB, dC, m, n, k);


}