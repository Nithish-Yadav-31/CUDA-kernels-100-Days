#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void Conv2D(float *input, float *kernel, int k, int n, float *output)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    float value = 0.0f;
    for (int i = -n; i <= n; i++){
        for(int j =-n; j <= n; j++){
            if(0 <= x + i && x + i < k && 0 <= y + j && y + j < k)
            {
                value += input[k * (y + j) + x + i] * kernel[(2 * n + 1) * (n + j) + n + j];
            }
        }
    }
    output[k * y + x] = value;
}


int main()
{
    int k = 32;
    int n = 2;
    int kernel_dim = 2 * n + 1;

    float *h_kernel = (float *)malloc(kernel_dim * kernel_dim * sizeof(float));
    for (int i = 0; i < kernel_dim; i++){
        for (int j = 0; j < kernel_dim; j++){
            h_kernel[i * kernel_dim + j] = 5.0f - abs((2 - i) + abs(2 - j));
        }
    }

    float *h_input = (float *)malloc(k * k * sizeof(float));
    for(int r = 0; r < k; r++){
        for(int c = 0; c < k; c++){
            h_input[r * k + c] = (float)(r + 1 + c);
        }
    }

    float *h_output = (float *)malloc(k * k * sizeof(float));

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, k * k * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernel_dim * kernel_dim * sizeof(float));
    cudaMalloc((void **)&d_output, k * k * sizeof(float));

    cudaMemcpy(d_input, h_input, k * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_dim * kernel_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(k, k);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    Conv2D<<<1, block_size>>>(d_input, d_kernel, k, n, d_output);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);

    cudaMemcpy(h_output, d_output, k * k * sizeof(float), cudaMemcpyDeviceToHost);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    printf("\nTiming Results:\n");
    printf("Kernel execution time: %.3f ms\n", kernel_time);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}
