#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024  // Number of elements in the array
#define THREADS_PER_BLOCK 256  // Number of threads per block

__global__ void tanh_kernel(float *input, float *output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size){
        output[idx] = tanh(input[idx]);
    }
}


int main()
{
    float *d_input, *d_output, *h_input, *h_output;
    size_t size = N * sizeof(float);

    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    for (int i = 0; i < N; i++){
        h_input[i] = (float)i - (N / 2);
    }

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blocks = ( N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    tanh_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        printf("tanh(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;

}
