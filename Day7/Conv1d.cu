#include <iostream>
#include <cuda_runtime.h>

#define N 16
#define F 3

__global__ void Conv1D(float *input, float *filter, float *output, int input_size, int filter_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < input_size - filter_size + 1){
        float result = 0.0f;
        for (int j = 0; j < filter_size; j++){
            result += input[i + j] * filter[j];
        }
        output[i] = result;
    }
}

int main()
{
    float input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float filter[F] = {0.25, 0.5, 0.25}; // Example filter (simple smoothing filter)
    float output[N - F + 1]; // Output size is input size minus filter size + 1

    float *d_input, *d_filter, *d_output;

    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_filter, N * sizeof(float));
    cudaMalloc((void **)&d_output, (N - F + 1) * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, F * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;
    int grid_size = (N - F + 1 + block_size -1 ) / block_size;
    Conv1D<<<grid_size, block_size>>>(d_input, d_filter, d_output, N, F);

    cudaMemcpy(output, d_output, (N - F +1 )* sizeof(float), cudaMemcpyDeviceToHost);
    // Print the result
    std::cout << "Input signal: ";
    for (int i = 0; i < N; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Filtered output: ";
    for (int i = 0; i < N - F + 1; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
