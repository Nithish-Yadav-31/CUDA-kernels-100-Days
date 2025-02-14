#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define THREADS_PER_BLOCK 256
#define PI 3.14f

__global__ void GeLU(float *input, float *output, int size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N){
        output[idx] = 0.5f * input[idx] * (1 + tanhf(sqrt(2.0f / PI) * (input[idx] + 0.44715f * powf(input[idx], 3)) ));
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
    

    GeLU<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        printf("GeLU(%f) = %f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
    

}
