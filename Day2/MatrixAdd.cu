#include <stdio.h>
#include <cuda_runtime.h>

#define N 100

__global__ void MatrixAdd(int *A, int *B, int *C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n){
        C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
}

int main()
{
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    h_A = (int *)malloc(N * N * sizeof(int));
    h_B = (int *)malloc(N * N * sizeof(int));
    h_C = (int *)malloc(N * N * sizeof(int));

    for(int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_A[i * N + j] = i + j;
            h_B[i * N + j] = i * j;
        }
    }

    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4);
    dim3 gridDim((N + blockDim.x  - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    MatrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf(" MATRIX ADDITION \n");
    for (int i = 0; i < N; i ++){
        for (int j = 0; j < N; j++){
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
