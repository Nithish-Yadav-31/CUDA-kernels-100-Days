#include <stdio.h>
#include <cuda_runtime.h>

const int N = 5;

__global__ void MatrixVecMul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * B[j]; 
        }
        C[i] = sum; 
    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(N * N * sizeof(float));
    h_B = (float *)malloc(N * sizeof(float));
    h_C = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        h_B[i] = (float)(i + 1);
        for(int j = 0; j < N; j++){
            h_A[i * N + j] = (float)(i + j);
        }
    }

    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16);
    dim3 gridDim((N + blockDim.x - 1)/ blockDim.x);

    MatrixVecMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result vector C
    printf("Matrix-Vector Multiplication (y = A * x):\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_C[i]);
    }
    printf("\n");

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}
