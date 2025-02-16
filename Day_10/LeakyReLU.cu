#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void Leaky_ReLU(float *input, float *output, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N){
        output[idx] = fmaxf(0.01f * input[idx], input[idx]);
    }
}

int main()
{
    float *d_input, *h_input, *d_output, *h_output;
    size_t size = N * sizeof(float);

    h_input = (float *)malloc(size);
    h_output = (float *)malloc(size);

    for (int i = 0; i < N; i++){
        h_input[i] = (float)i - N / 2;
    }

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int block_size = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    Leaky_ReLU<<<block_size, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        printf("LeakyReLU(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
    

}
