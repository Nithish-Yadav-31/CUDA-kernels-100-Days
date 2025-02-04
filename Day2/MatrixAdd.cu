#include <iostream>
#include <cuda_runtime.h> // Include CUDA runtime library
#include <cmath> // For ceil function
#include <cstdio>  // For printf


// Kernel A : iterates rows  within each thread
__global__ void MatrixAdd_A(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index

    if (i < N) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}


// Kernel B : 2D kernel. Each thread calculates 1 cell
__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index

    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}


// Kernel D:  iterates columns  within each thread

__global__ void MatrixAdd_D(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index

    if (j < N) {
        for (int i = 0; i < N; i++) {
            C[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
}



int main() {
    const int N = 10;
    float *A, *B, *C;
    float *d_a, *d_b, *d_c; // Device pointers
    cudaError_t cudaStatus;


    // Host Memory Allocation and Initialization
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        std::cerr << "Host memory allocation failed!\n";
        return 1;  // Indicate an error
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;  // Initialize C to zero before using GPU computation.  Important!
        }
    }

    // Device Memory Allocation
    cudaStatus = cudaMalloc((void **)&d_a, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_a failed: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }

    cudaStatus = cudaMalloc((void **)&d_b, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_b failed: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }
    cudaStatus = cudaMalloc((void **)&d_c, N * N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc d_c failed: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }



    // Host to Device Memory Copy
    cudaStatus = cudaMemcpy(d_a, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy d_a failed: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }
    cudaStatus = cudaMemcpy(d_b, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy d_b failed: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }


   // Launch kernel B
    dim3 dimBlockB(16, 16); // Define block dimensions for kernel B
    dim3 dimGridB(ceil((float)N / dimBlockB.x), ceil((float)N / dimBlockB.y));  // Ensure full coverage


    MatrixAdd_B<<<dimGridB, dimBlockB>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
                std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
                return 1;
    }

    cudaStatus = cudaDeviceSynchronize(); // Wait for kernel to complete
      if (cudaStatus != cudaSuccess) {
                std::cerr << "Device synchronize failed: " << cudaGetErrorString(cudaStatus) << "\n";
                return 1;
    }


    // Device to Host Memory Copy
    cudaStatus = cudaMemcpy(C, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpy C failed: " << cudaGetErrorString(cudaStatus) << "\n";
                return 1;
    }



    // Print Result Matrix C
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }


    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(A);
    free(B);
    free(C);

    return 0;
}
