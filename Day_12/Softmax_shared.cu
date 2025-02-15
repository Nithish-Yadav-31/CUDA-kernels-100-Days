#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 5
#define THREADS_PER_BLOCK 256

__global__ void Softmax(float *input, float *output, int size){
    extern __shared__ float shared_exp[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < size){
        shared_exp[threadIdx.x] = expf(input[idx]);
    }else{
        shared_exp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (threadIdx.x < stride){
            shared_exp[threadIdx.x] += shared_exp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (idx < size){
        output[idx] = shared_exp[threadIdx.x] / shared_exp[0];
    }
}




int main()
{
    float *h_input, *d_input, *h_output, *d_output;
    size_t size = N * sizeof(float);
    cudaEvent_t start, stop;
    float elapsedTime;

    h_input = (float *)malloc(size);
    h_output = (float *)malloc(size);

    for (int i = 0; i < N; i++){
        h_input[i] = ((float)i - N / 2.0f) / N;
    }

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // Launch kernel with shared memory allocation
    Softmax<<<num_blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_output, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Verify results
    for (int i = 0; i < N; i++) {
        printf("Softmax(%f) = %f\n", h_input[i], h_output[i]);
    }

    printf("Total execution time: %f ms\n", elapsedTime);

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
    

}
