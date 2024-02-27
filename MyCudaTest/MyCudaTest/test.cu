#include"test.cuh"
using namespace std;

void testGEMM() {
	int m = 2, n = 4, k = 3;
	int sizeA = sizeof(float) * m * k;
	int sizeB = sizeof(float) * k * n;
	int sizeC = sizeof(float) * m * n;
	float* A = (float*)malloc(sizeA);
	float* B = (float*)malloc(sizeB);
	float* C = (float*)malloc(sizeC);

	float* dA, * dB, * dC;
	cudaMalloc((void**)&dA, sizeA);
	cudaMalloc((void**)&dB, sizeB);
	cudaMalloc((void**)&dC, sizeC);

	for (int i = 0; i < m * k; i++) {
		A[i] = i;
		//printf("A%d = %f ", i, A[i]);
	}
	cout << endl;
	for (int i = 0; i < k * n; i++) {
		B[i] = 1.0;
	}

	cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);


	cublasHandle_t base_handle;
	cublasCreate(&base_handle);
	float alpa = 1.0, beta = 0.0;
	/*cublasSgemm(base_handle, CUBLAS_OP_N, CUBLAS_OP_N,
		n, m, k, &alpa,
		dB, n, dA, k, &beta, dC, n
	);*/
	cublasSgemm(base_handle, CUBLAS_OP_T, CUBLAS_OP_T,
		m, n, k, &alpa,
		dA, k, dB, n, &beta, dC, m
	);
	cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < m * n; i++) {
		printf("%d %f\n", i, C[i]);
	}
}

void testGEMV() {
	int m = 2, n = 4;
	int sizeA = sizeof(float) * m * n;
	int sizeB = sizeof(float) * n;
	int sizeC = sizeof(float) * m;

	float* A = (float*)malloc(sizeA);
	float* B = (float*)malloc(sizeB);
	float* C = (float*)malloc(sizeC);
	float* dA, * dB, * dC;
	cudaMalloc((void**)&dA, sizeA);
	cudaMalloc((void**)&dB, sizeB);
	cudaMalloc((void**)&dC, sizeC);

	for (int i = 0; i < m * n; i++) {
		A[i] = i;

	}

	for (int i = 0; i < n; i++) {
		B[i] = 1.0;
	}
	for (int i = 0; i < m; i++) {
		C[i] = 0.0;
	}
	cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);


	cublasHandle_t base_handle;
	cublasCreate(&base_handle);
	float alpa = 1.0, beta = 0.0;

	cublasSgemv(base_handle, CUBLAS_OP_T,
		n, m, &alpa,
		dA, n, dB, 1, &beta, dC, 1);

	cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < m; i++) {
		printf("%d %f\n", i, C[i]);
	}
}
__global__ void testDiv() {
	int tid = threadIdx.x;
	if (tid < 32) {
		printf("%d \n", tid);
	}
	else {
		printf("%d \n", tid);
	}
}
template<const int num_per_thread, const int bin>
__global__ void testHistogram(int* a, int* y, const int N) {


}

template<const int thread_per_block>
__global__ void inclusiveScan(float* in, float* out, int N) {
	__shared__ float s_tmp[thread_per_block];
	int tid = threadIdx.x;
	s_tmp[tid] = in[blockIdx.x * blockDim.x + tid];
	__syncthreads();
	for (int i = 1; i < N; i <<= 1) {
		if (tid >= i) {
			s_tmp[tid] += s_tmp[tid - i];
		}
		__syncthreads();
	}
	out[blockIdx.x * blockDim.x + tid] = s_tmp[tid];

}
void test() {
	testGEMV();
}