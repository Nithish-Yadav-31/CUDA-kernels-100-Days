#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Size of the tile (block), typically a power of 2 like 16, 32, etc.

__global__ void matmul_kernel(float *A, float *B, float *C, int N) {
    // Shared memory to store a TILE_SIZE x TILE_SIZE block of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread's row and column in the resulting matrix C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over the tiles of matrix A and B
    for (int t = 0; t < (N / TILE_SIZE); t++) {
        // Load one tile of A into shared memory
        // Note: A is accessed row-wise in memory, so each thread loads one element from the row
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];

        // Load one tile of B into shared memory
        // Note: B is accessed column-wise in memory, so each thread loads one element from the column
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // Synchronize threads to ensure all threads have finished loading the tiles into shared memory
        __syncthreads();

        // Perform the multiplication for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize threads again before loading the next tile (important for correct shared memory usage)
        __syncthreads();
    }

    // Store the result in the C matrix if within bounds
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

void matmul(float *A, float *B, float *C, int N) {
    float *d_A, *d_B, *d_C;

    // Step 1: Allocate memory on the GPU for matrices A, B, and C
    // This is where the matrix data will be stored during computation
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Step 2: Copy matrices A and B from host memory to device memory
    // We need to copy the data because CUDA works with data stored on the GPU (device)
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Step 3: Set up the execution configuration for the kernel
    // Number of threads per block (blockSize) is TILE_SIZE x TILE_SIZE
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    
    // Number of blocks in the grid (gridSize) is N / TILE_SIZE for both x and y dimensions
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    // Step 4: Launch the kernel with the specified grid and block sizes
    // Each block computes one TILE_SIZE x TILE_SIZE block of the output matrix C
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Step 5: Copy the result matrix C from device to host memory
    // After the kernel completes, the result is in d_C on the device,
    // and we need to copy it back to the host (CPU) for further use
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 6: Free the memory allocated on the GPU for matrices A, B, and C
    // It's important to clean up to avoid memory leaks
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 256;  // Size of the matrix (N x N)

    // Allocate memory for matrices A, B, and C on the host (CPU)
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));

    // Step 1: Initialize matrices A and B with random values
    // This is for testing purposes, so we just fill them with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand() % 100);  // Random values between 0 and 99
        B[i] = static_cast<float>(rand() % 100);  // Random values between 0 and 99
    }

    // Step 2: Perform matrix multiplication (A * B = C)
    // The result will be stored in matrix C
    matmul(A, B, C, N);

    // Optionally, print the result matrix C for smaller matrices
    // We do not usually print large matrices for performance reasons
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Step 3: Free the memory allocated for matrices A, B, and C on the host
    free(A);
    free(B);
    free(C);

    return 0;
}
