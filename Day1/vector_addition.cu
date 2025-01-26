#include <iostream>
#include <cmath>

// CUDA kernel function to add two vectors
__global__ void addVectors(const float* A, const float* B, float* C, int N){
    // Calculate the global thead index based on block and thread indices.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    const int N = 10; // Defining the size of vectors
    float A[N], B[N], C[N]; // Arrays to hold the vectors and results

    // Initializing vectors with some values
    for(int i =0; i < N; ++i){
        A[i] = i+1;
        B[i] = (i+1) * 2;
    }

    // initializing device pointer of vectors A, B and C
    float *d_a, *d_b, *d_c;
    
    // Allocate memory on the GPU for the device pointer
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data from host arrays to device arrays
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size 
    int blocksize = 256; 

    // Calculating the gridsize to make sure enough blocks are created 
    int gridsize = (N + blocksize - 1) / blocksize;

    // Launch the kernel with the specified grid and block dimensions
    addVectors<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

    // Copy from device to host vectors
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyHostToDevice);

    // Printing the result
    for (int i = 0; i < N; ++i){
        std::cout<< "c[" << i << "]" << C[i] << std::endl;
    }

    // Free the allocated memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;


}