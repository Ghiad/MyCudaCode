#include"GEMM.cuh"
using namespace std;

#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define CHECK_CUDA_ERROR(fun) \
{	\
	cudaError_t err = (fun);	\
	if (err != cudaSuccess)		\
		printf("%s %d CUDA:%s", __FILE__, __LINE__, cudaGetErrorString(err));	\
}


template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K,
	const int THREAD_SIZE_Y,
	const int THREAD_SIZE_X
>
__global__ void gemm_v0(float* A, float* B, float* C, int M, int N, int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	const int tid = tx + ty * blockDim.x;

	__shared__ float s_a[BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
	float r_a[THREAD_SIZE_Y];
	float r_b[THREAD_SIZE_X];
	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0f };

	const int THREAD_PER_BLOCK = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_SIZE_X * THREAD_SIZE_Y);
	const int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
	int A_START_ROW = tid / A_THREAD_PER_ROW;
	int A_START_COL = (tid % A_THREAD_PER_ROW) * 4;
	const int A_STRIDE = THREAD_PER_BLOCK / A_THREAD_PER_ROW;

	const int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
	int B_START_ROW = tid / B_THREAD_PER_ROW;
	int B_START_COL = (tid % B_THREAD_PER_ROW) * 4;
	const int B_STRIDE = THREAD_PER_BLOCK / B_THREAD_PER_ROW;

	const int ldg_num = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (4 * THREAD_PER_BLOCK);
	float ldg_a[4 * ldg_num];

	A = &A[by * K * BLOCK_SIZE_M];
	B = &B[bx * BLOCK_SIZE_N];
#pragma unroll
	for (int i = 0; i < K; i += BLOCK_SIZE_K) {
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_M; j += A_STRIDE) {
			int ldg_index = (j / A_STRIDE) * 4;
			FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[i + (A_START_ROW + j) * K + A_START_COL]);
			s_a[A_START_COL][A_START_ROW + j] = ldg_a[ldg_index];
			s_a[A_START_COL + 1][A_START_ROW + j] = ldg_a[1 + ldg_index];
			s_a[A_START_COL + 2][A_START_ROW + j] = ldg_a[2 + ldg_index];
			s_a[A_START_COL + 3][A_START_ROW + j] = ldg_a[3 + ldg_index];
		}
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_K; j += B_STRIDE) {
			FETCH_FLOAT4(s_b[B_START_ROW + j][B_START_COL]) = FETCH_FLOAT4(B[N * (i + B_START_ROW + j) + B_START_COL]);
		}
		__syncthreads();
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE_K; k++) {
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_Y; j += 4) {
				FETCH_FLOAT4(r_a[j]) = FETCH_FLOAT4(s_a[k][ty * THREAD_SIZE_Y + j]);
			}
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_X; j += 4) {
				FETCH_FLOAT4(r_b[j]) = FETCH_FLOAT4(s_b[k][tx * THREAD_SIZE_X + j]);
			}
#pragma unroll
			for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
				for (int n = 0; n < THREAD_SIZE_X; n++) {
					accum[m][n] += r_a[m] * r_b[n];
				}
			}
		}

		__syncthreads();

	}
	//C = &C[blockIdx.y * N * BLOCK_SIZE_M + blockIdx.x * BLOCK_SIZE_N];
#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
		for (int n = 0; n < THREAD_SIZE_X; n += 4) {

			/*FETCH_FLOAT4(C[(ty * THREAD_SIZE_Y + by * BLOCK_SIZE_M + j) * N + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + l])
				= FETCH_FLOAT4(accum[j][l]);*/
			FETCH_FLOAT4(C[N * (by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + m) + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + n])
				= FETCH_FLOAT4(accum[m][n]);
		}
	}
}


template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K,
	const int THREAD_SIZE_Y,
	const int THREAD_SIZE_X
>
__global__ void gemm_v1(float* A, float* B, float* C, const int M, const int N, const int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	const int tid = tx + ty * blockDim.x;

	const int threads_per_block = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_SIZE_Y * THREAD_SIZE_X);
	const int a_threads_per_row = BLOCK_SIZE_K / 4;
	const int a_start_row = tid / a_threads_per_row;
	const int a_start_col = tid % a_threads_per_row * 4;
	const int a_stride = threads_per_block / a_threads_per_row;
	const int a_ldg_num = (BLOCK_SIZE_M * BLOCK_SIZE_K) / (4 * threads_per_block);

	const int b_threads_per_row = BLOCK_SIZE_N / 4;
	const int b_start_row = tid / b_threads_per_row;
	const int b_start_col = tid % b_threads_per_row * 4;
	const int b_stride = threads_per_block / b_threads_per_row;
	const int b_ldg_num = (BLOCK_SIZE_N * BLOCK_SIZE_K) / (4 * threads_per_block);
	/*const int a_stride = BLOCK_SIZE_M / a_threads_per_row;
	const int b_stride = BLOCK_SIZE_K / b_threads_per_row;
	const int a_ldg_num = (BLOCK_SIZE_M * BLOCK_SIZE_K) / 4 * threads_per_block;
	const int b_ldg_num = (BLOCK_SIZE_N * BLOCK_SIZE_K) / 4 * threads_per_block;*/
	//const int b_ldg_num = (BLOCK_SIZE_M * BLOCK_SIZE_K) / 4 * threads_per_block;

	__shared__ float s_a[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float s_b[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
	float r_a[2][THREAD_SIZE_Y];
	float r_b[2][THREAD_SIZE_X];
	float ldg_a[4 * a_ldg_num];
	//没写ldg_b
	float ldg_b[4 * b_ldg_num];
	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0f };

	int write_idx = 0;

	A = &A[by * K * BLOCK_SIZE_M];
	B = &B[bx * BLOCK_SIZE_N];
	/*A = &A[by * K * blockDim.y];
	B = &B[bx * blockDim.x];*/
#pragma unroll
	for (int i = 0; i < BLOCK_SIZE_M; i += a_stride) {
		int ldg_index = i / a_stride * 4;
		FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[(a_start_row + i) * K + a_start_col]);
		s_a[write_idx][a_start_col][a_start_row + i] = ldg_a[ldg_index];
		s_a[write_idx][a_start_col + 1][a_start_row + i] = ldg_a[ldg_index + 1];
		s_a[write_idx][a_start_col + 2][a_start_row + i] = ldg_a[ldg_index + 2];
		s_a[write_idx][a_start_col + 3][a_start_row + i] = ldg_a[ldg_index + 3];
	}
#pragma unroll
	for (int i = 0; i < BLOCK_SIZE_K; i += b_stride) {
		FETCH_FLOAT4(s_b[write_idx][b_start_row + i][b_start_col]) = FETCH_FLOAT4(B[(b_start_row + i) * N + b_start_col]);
	}
	__syncthreads();

	//FETCH_FLOAT4(r_a[write_idx][0]) = FETCH_FLOAT4(s_a[write_idx][0][ty]);
	//FETCH_FLOAT4(r_b[write_idx][0]) = FETCH_FLOAT4(s_b[write_idx][0][0]);
#pragma unroll
	for (int i = 0; i < THREAD_SIZE_Y; i += 4) {
		FETCH_FLOAT4(r_a[0][i]) = FETCH_FLOAT4(s_a[0][0][ty * THREAD_SIZE_Y + i]);
	}
#pragma unroll
	for (int i = 0; i < THREAD_SIZE_X; i += 4) {
		FETCH_FLOAT4(r_b[0][i]) = FETCH_FLOAT4(s_b[0][0][tx * THREAD_SIZE_X + i]);
	}
	write_idx ^= 1;
#pragma unroll
	for (int i = 0; i < K; ) {
		i += BLOCK_SIZE_K;
		if (i < K) {
			//先存到ldg中，后面再存回s_a
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_M; j += a_stride) {
				int ldg_index = (j / a_stride) * 4;
				//FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[(a_start_row + j) * K + a_start_col]);
				FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[(a_start_row + j) * K + a_start_col + i]);
				/*s_a[write_idx][a_start_col][a_start_row + j] = ldg_a[ldg_index];
				s_a[write_idx][a_start_col + 1][a_start_row + j] = ldg_a[ldg_index + 1];
				s_a[write_idx][a_start_col + 2][a_start_row + j] = ldg_a[ldg_index + 2];
				s_a[write_idx][a_start_col + 3][a_start_row + j] = ldg_a[ldg_index + 3];*/
			}
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_K; j += b_stride) {
				/*FETCH_FLOAT4(s_b[write_idx][b_start_row + j][b_start_col]) = FETCH_FLOAT4(B[(b_start_row + j) * N + b_start_col]);*/
				int ldg_index = (j / b_stride) * 4;
				//int ldg_index = (j / a_stride) * 4;
				//FETCH_FLOAT4(ldg_b[ldg_index]) = FETCH_FLOAT4(B[(b_start_row + j) * N + b_start_col]);
				FETCH_FLOAT4(ldg_b[ldg_index]) = FETCH_FLOAT4(B[(b_start_row + j + i) * N + b_start_col]);
			}
		}
		//小迭代预取出错
		/*for (int k = 0; k < BLOCK_SIZE_K; k++) {
			for (int j = 0; j < THREAD_SIZE_Y; j += 4) {
				FETCH_FLOAT4(r_a[write_idx][j]) = FETCH_FLOAT4(s_a[load_idx][k][j + ty * THREAD_SIZE_Y]);
			}
			for (int j = 0; j < THREAD_SIZE_X; j += 4) {
				FETCH_FLOAT4(r_b[write_idx][j]) = FETCH_FLOAT4(s_b[load_idx][k][j + tx * THREAD_SIZE_X]);
			}

			for (int m = 0; m < THREAD_SIZE_Y; m++) {
				for (int n = 0; n < THREAD_SIZE_X; n++) {
					accum[m][n] += r_a[write_idx^1][m] * r_b[write_idx^1][n];
				}
			}

		}*/
		int load_idx = write_idx ^ 1;
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_K - 1; j++) {
#pragma unroll
			for (int k = 0; k < THREAD_SIZE_Y; k += 4) {
				//FETCH_FLOAT4(r_a[(k + 1) % 2][j]) = FETCH_FLOAT4(s_a[load_idx][k][j + ty * THREAD_SIZE_Y]);
				FETCH_FLOAT4(r_a[(j + 1) % 2][k]) = FETCH_FLOAT4(s_a[load_idx][j + 1][k + ty * THREAD_SIZE_Y]);
			}
#pragma unroll
			for (int k = 0; k < THREAD_SIZE_X; k += 4) {
				FETCH_FLOAT4(r_b[(j + 1) % 2][k]) = FETCH_FLOAT4(s_b[load_idx][j + 1][k + tx * THREAD_SIZE_X]);
				//FETCH_FLOAT4(r_b[(k + 1) % 2][j]) = FETCH_FLOAT4(s_b[load_idx][k][j + tx * THREAD_SIZE_X]);

			}
#pragma unroll
			for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
				for (int n = 0; n < THREAD_SIZE_X; n++) {
					accum[m][n] += r_a[j % 2][m] * r_b[j % 2][n];
				}
			}
		}

		if (i < K) {
			//先存回s_a，再进行最后一次小迭代
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_M; j += a_stride) {
				int ldg_index = (j / a_stride) * 4;
				s_a[write_idx][a_start_col][a_start_row + j] = ldg_a[ldg_index];
				s_a[write_idx][a_start_col + 1][a_start_row + j] = ldg_a[ldg_index + 1];
				s_a[write_idx][a_start_col + 2][a_start_row + j] = ldg_a[ldg_index + 2];
				s_a[write_idx][a_start_col + 3][a_start_row + j] = ldg_a[ldg_index + 3];
			}
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_K; j += b_stride) {
				/*FETCH_FLOAT4(s_b[write_idx][b_start_row + j][b_start_col]) = FETCH_FLOAT4(B[(b_start_row + j) * N + b_start_col]);*/
				int ldg_index = (j / b_stride) * 4;
				FETCH_FLOAT4(s_b[write_idx][b_start_row + j][b_start_col]) = FETCH_FLOAT4(ldg_b[ldg_index]);
			}
			__syncthreads();
			write_idx ^= 1;
		}


		//最后一次小迭代
#pragma unroll
		for (int j = 0; j < THREAD_SIZE_Y; j += 4) {
			FETCH_FLOAT4(r_a[0][j]) = FETCH_FLOAT4(s_a[load_idx ^ 1][0][j + ty * THREAD_SIZE_Y]);
			//FETCH_FLOAT4(r_a[0][j]) = FETCH_FLOAT4(s_a[load_idx][BLOCK_SIZE_K - 1][j + ty * THREAD_SIZE_Y]);
		}
#pragma unroll
		for (int j = 0; j < THREAD_SIZE_X; j += 4) {
			FETCH_FLOAT4(r_b[0][j]) = FETCH_FLOAT4(s_b[load_idx ^ 1][0][j + tx * THREAD_SIZE_X]);
		}
#pragma unroll
		for (int m = 0; m < THREAD_SIZE_Y; m++) {
			for (int n = 0; n < THREAD_SIZE_X; n++) {
				accum[m][n] += r_a[1][m] * r_b[1][n];
			}
		}
	}
#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
		for (int n = 0; n < THREAD_SIZE_X; n += 4) {
			/*FETCH_FLOAT4(C[N * (by * blockDim.y + ty + m) + bx * blockDim.x + tx + n])
				= FETCH_FLOAT4(accum[m][n]);*/
			FETCH_FLOAT4(C[N * (by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + m) + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + n])
				= FETCH_FLOAT4(accum[m][n]);
		}
		//for (int n = 0; n < THREAD_SIZE_X; n++) {
		//	/*FETCH_FLOAT4(C[N * (by * blockDim.y + ty + m) + bx * blockDim.x + tx + n])
		//		= FETCH_FLOAT4(accum[m][n]);*/
		//	FETCH_FLOAT4(C[N * (by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + m) + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + n])
		//		= FETCH_FLOAT4(accum[m][n]);
		//}
	}
}


template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K,
	const int THREAD_SIZE_Y,
	const int THREAD_SIZE_X
>
__global__ void gemm_v2(float* A, float* B, float* C, const int M, const int N, const int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	const int tid = tx + ty * blockDim.x;

	const int threads_per_block = (BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_SIZE_Y * THREAD_SIZE_X);
	const int a_threads_per_row = BLOCK_SIZE_K / 4;
	const int a_start_row = tid / a_threads_per_row;
	const int a_start_col = tid % a_threads_per_row * 4;
	const int a_stride = threads_per_block / a_threads_per_row;
	const int a_ldg_num = (BLOCK_SIZE_M * BLOCK_SIZE_K) / (4 * threads_per_block);

	const int b_threads_per_row = BLOCK_SIZE_N / 4;
	const int b_start_row = tid / b_threads_per_row;
	const int b_start_col = tid % b_threads_per_row * 4;
	const int b_stride = threads_per_block / b_threads_per_row;
	const int b_ldg_num = (BLOCK_SIZE_N * BLOCK_SIZE_K) / (4 * threads_per_block);

	__shared__ float s_a[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
	__shared__ float s_b[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
	float r_a[2][THREAD_SIZE_Y];
	float r_b[2][THREAD_SIZE_X];
	float ldg_a[4 * a_ldg_num];
	float ldg_b[4 * b_ldg_num];
	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0.0f };


	A = &A[by * K * BLOCK_SIZE_M];
	B = &B[bx * BLOCK_SIZE_N];

#pragma unroll
	for (int i = 0; i < BLOCK_SIZE_M; i += a_stride) {
		int ldg_index = i / a_stride * 4;
		FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_start_row + i, 
			a_start_col, 
			K)]);
		s_a[0][a_start_col][a_start_row + i] = ldg_a[ldg_index];
		s_a[0][a_start_col + 1][a_start_row + i] = ldg_a[ldg_index + 1];
		s_a[0][a_start_col + 2][a_start_row + i] = ldg_a[ldg_index + 2];
		s_a[0][a_start_col + 3][a_start_row + i] = ldg_a[ldg_index + 3];
	}
#pragma unroll
	for (int i = 0; i < BLOCK_SIZE_K; i += b_stride) {
		FETCH_FLOAT4(s_b[0][b_start_row + i][b_start_col]) = FETCH_FLOAT4(B[OFFSET(b_start_row+i,
			b_start_col,
			N)]);
	}
	__syncthreads();


	/*忘了 /2 s_b也写错了
	FETCH_FLOAT4(r_a[0][0]) = FETCH_FLOAT4(s_a[0][0][ty * THREAD_SIZE_Y]);
	FETCH_FLOAT4(r_a[0][4]) = FETCH_FLOAT4(s_a[0][0][ty * THREAD_SIZE_Y + BLOCK_SIZE_M / 2]);
	FETCH_FLOAT4(r_b[0][0]) = FETCH_FLOAT4(s_a[0][0][tx * THREAD_SIZE_X]);
	FETCH_FLOAT4(r_b[0][4]) = FETCH_FLOAT4(s_a[0][0][tx * THREAD_SIZE_X + BLOCK_SIZE_N / 2]);*/
	FETCH_FLOAT4(r_a[0][0]) = FETCH_FLOAT4(s_a[0][0][ty * THREAD_SIZE_Y / 2]);
	FETCH_FLOAT4(r_a[0][4]) = FETCH_FLOAT4(s_a[0][0][ty * THREAD_SIZE_Y / 2 + 64]);
	FETCH_FLOAT4(r_b[0][0]) = FETCH_FLOAT4(s_b[0][0][tx * THREAD_SIZE_X / 2]);
	FETCH_FLOAT4(r_b[0][4]) = FETCH_FLOAT4(s_b[0][0][tx * THREAD_SIZE_X / 2 + 64]);

	int write_idx = 1;
#pragma unroll
	for (int i = 0; i < K; ) {
		i += BLOCK_SIZE_K;
		if (i < K) {

#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_M; j += a_stride) {
				int ldg_index = (j / a_stride) * 4;
				FETCH_FLOAT4(ldg_a[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_start_row+j,a_start_col+i,K)]);
			}
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_K; j += b_stride) {
				int ldg_index = (j / b_stride) * 4;
				FETCH_FLOAT4(ldg_b[ldg_index]) = FETCH_FLOAT4(B[OFFSET(b_start_row+j+i,b_start_col,N)]);
			}
		}

		int load_idx = write_idx ^ 1;
#pragma unroll
		for (int j = 0; j < BLOCK_SIZE_K - 1; j++) {

			FETCH_FLOAT4(r_a[(j + 1) % 2][0]) = FETCH_FLOAT4(s_a[load_idx][j + 1][ty * THREAD_SIZE_Y / 2]);
			FETCH_FLOAT4(r_a[(j + 1) % 2][4]) = FETCH_FLOAT4(s_a[load_idx][j + 1][ty * THREAD_SIZE_Y / 2 + 64]);
			FETCH_FLOAT4(r_b[(j + 1) % 2][0]) = FETCH_FLOAT4(s_b[load_idx][j + 1][tx * THREAD_SIZE_X / 2]);
			FETCH_FLOAT4(r_b[(j + 1) % 2][4]) = FETCH_FLOAT4(s_b[load_idx][j + 1][tx * THREAD_SIZE_X / 2 + 64]);

#pragma unroll
			for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
				for (int n = 0; n < THREAD_SIZE_X; n++) {
					accum[m][n] += r_a[j % 2][m] * r_b[j % 2][n];
				}
			}
		}

		if (i < K) {
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_M; j += a_stride) {
				int ldg_index = (j / a_stride) * 4;
				s_a[write_idx][a_start_col][a_start_row + j] = ldg_a[ldg_index];
				s_a[write_idx][a_start_col + 1][a_start_row + j] = ldg_a[ldg_index + 1];
				s_a[write_idx][a_start_col + 2][a_start_row + j] = ldg_a[ldg_index + 2];
				s_a[write_idx][a_start_col + 3][a_start_row + j] = ldg_a[ldg_index + 3];
			}
#pragma unroll
			for (int j = 0; j < BLOCK_SIZE_K; j += b_stride) {
				int ldg_index = (j / b_stride) * 4;
				FETCH_FLOAT4(s_b[write_idx][b_start_row + j][b_start_col]) = FETCH_FLOAT4(ldg_b[ldg_index]);
			}
			__syncthreads();
			write_idx ^= 1;
		}


		//最后一次小迭代
		FETCH_FLOAT4(r_a[0][0]) = FETCH_FLOAT4(s_a[load_idx ^ 1][0][ty * THREAD_SIZE_Y / 2]);
		FETCH_FLOAT4(r_a[0][4]) = FETCH_FLOAT4(s_a[load_idx ^ 1][0][ty * THREAD_SIZE_Y / 2 + 64]);
		FETCH_FLOAT4(r_b[0][0]) = FETCH_FLOAT4(s_b[load_idx ^ 1][0][tx * THREAD_SIZE_X / 2]);
		FETCH_FLOAT4(r_b[0][4]) = FETCH_FLOAT4(s_b[load_idx ^ 1][0][tx * THREAD_SIZE_X / 2 + 64]);

#pragma unroll
		for (int m = 0; m < THREAD_SIZE_Y; m++) {
#pragma unroll
			for (int n = 0; n < THREAD_SIZE_X; n++) {
				accum[m][n] += r_a[1][m] * r_b[1][n];
			}
		}
	}
#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y / 2; m++) {
		FETCH_FLOAT4(C[OFFSET(
			by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y / 2 + m,
			bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X / 2,
			N)]) = FETCH_FLOAT4(accum[m][0]);
	}

#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y / 2; m++) {
		FETCH_FLOAT4(C[OFFSET(
			by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y / 2 + m,
			bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X / 2 + 64,
			N)]) = FETCH_FLOAT4(accum[m][4]);
	}
#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y / 2; m++) {
		FETCH_FLOAT4(C[OFFSET(
			by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y / 2 + m + 64,
			bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X / 2,
			N)]) = FETCH_FLOAT4(accum[m + 4][0]);

	}
#pragma unroll
	for (int m = 0; m < THREAD_SIZE_Y / 2; m++) {

		FETCH_FLOAT4(C[OFFSET(
			by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y / 2 + m + 64,
			bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X / 2 + 64,
			N)]) = FETCH_FLOAT4(accum[m + 4][4]);

	}
}

void gemm() {
	const int m = 2048;
	const int n = 2048;
	const int k = 2048;
	//数据空间分配
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

	for (int i = 0; i < m * k; i++) {
		A[i] = i / 13;
	}
	for (int i = 0; i < n * k; i++) {
		B[i] = i % 13;
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i * k + j] = 0.0;
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

	//test << <1, dim3(5, 2) >> > (dC);
	//cudaDeviceSynchronize();

	//运行核函数
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float msecTotal = 0;
	int nIter = 1000;
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		gemm_v1<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X> << <gridsize, blocksize >> > (dA, dB, dC, m, n, k);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost);

	msecPerMatrixMul[0] = msecTotal / nIter;
	gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
	printf("My gemm_v0 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
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