#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

const int N = 10;

__global__ void VectorNorm(float* d_V, float* d_V_norm, int N, float magnitude){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        d_V_norm[i] = d_V[i] / magnitude;
    }
}


int main()
{
    float *h_V, *h_V_norm;
    float *d_V, *d_V_norm;

    h_V = (float *)malloc(N * sizeof(float));
    h_V_norm = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        h_V[i] = (float)(i+1);
    }

    cudaMalloc((void **)&d_V, N * sizeof(float));
    cudaMalloc((void **)&d_V_norm, N * sizeof(float));

    cudaMemcpy(d_V, h_V, N * sizeof(float), cudaMemcpyHostToDevice);

    float magnitude = 0.0f;
    for (int i = 0; i < N; i++){
        magnitude += h_V[i] * h_V[i];
    }
    magnitude= sqrt(magnitude);

    dim3 blockDim(16);
    dim3 gridDim((N * blockDim.x - 1) / blockDim.x);

    VectorNorm<<<gridDim, blockDim>>>(d_V, d_V_norm, N, magnitude);

    cudaMemcpy(h_V_norm, d_V_norm, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Print the normalized vector
    printf("Normalized Vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_V_norm[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_V);
    cudaFree(d_V_norm);

    // Free host memory
    free(h_V);
    free(h_V_norm);

    return 0;
}
