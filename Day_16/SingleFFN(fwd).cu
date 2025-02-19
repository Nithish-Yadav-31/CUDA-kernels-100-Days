#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function for forward pass of a single layer FFN
__global__ void forward_pass(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = sum + bias[idx];
    }
}


int main() {
    int input_size = 4;   // Number of input features
    int output_size = 3;  // Number of output neurons

    // Host memory allocation
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_weights = (float*)malloc(input_size * output_size * sizeof(float));
    float *h_bias = (float*)malloc(output_size * sizeof(float));
    float *h_output = (float*)malloc(output_size * sizeof(float));


    // Initialize host data (example values)
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (float)i + 1.0f;  // Example input
    }

    for (int i = 0; i < input_size * output_size; i++) {
        h_weights[i] = (float)(i + 1.0f) * 0.1f; // Example weights
    }

    for (int i = 0; i < output_size; i++) {
        h_bias[i] = (float)i * 0.5f; // Example bias
    }


    // Device memory allocation
    float *d_input;
    float *d_weights;
    float *d_bias;
    float *d_output;

    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_bias, output_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threads_per_block = 256;  // Adjust based on your GPU and output size
    int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;  // Calculate number of blocks

    // Launch the kernel
    forward_pass<<<num_blocks, threads_per_block>>>(d_input, d_weights, d_bias, d_output, input_size, output_size);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    printf("Output:\n");
    for (int i = 0; i < output_size; i++) {
        printf("Output[%d]: %f\n", i, h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_weights);
    free(h_bias);
    free(h_output);

    return 0;
}
