#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void Sigmoid(float *input, float *output, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < N){
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}


int main()
{
    float *h_input, *d_input, *h_output, *d_output;
    size_t size = N * sizeof(float);

    h_input = (float *)malloc(size);
    h_output = (float *)malloc(size);

    for (int i = 0; i < N; i++){
        h_input[i] = (float)i - N / 2.0f;
    }

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    

    Sigmoid<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        printf("Sigmoid(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
    

}
