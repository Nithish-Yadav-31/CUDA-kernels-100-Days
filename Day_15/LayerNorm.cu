#include <stdio.h>
#include <cuda_runtime.h>

// Fused Kernel for Layer Normalization (Mean, Variance, and Normalization)
__global__ void fused_layer_norm(float *input, float *output, float *mean, float *variance,
                                   float gamma, float beta, float epsilon, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Row index

    if (row < num_rows) {
        // 1. Calculate Mean
        float sum = 0.0f;
        for (int col = 0; col < num_cols; col++) {
            sum += input[row * num_cols + col];
        }
        mean[row] = sum / (float)num_cols;

        // 2. Calculate Variance
        float sum_sq_diff = 0.0f;
        for (int col = 0; col < num_cols; col++) {
            float diff = input[row * num_cols + col] - mean[row];
            sum_sq_diff += diff * diff;
        }
        variance[row] = sum_sq_diff / (float)num_cols;

        // 3. Normalize
        for (int col = 0; col < num_cols; col++) {
            float normalized_value = (input[row * num_cols + col] - mean[row]) / sqrtf(variance[row] + epsilon);
            output[row * num_cols + col] = gamma * normalized_value + beta;
        }
    }
}


int main() {
    int num_rows = 2;      // Number of rows (layers)
    int num_cols = 4;      // Number of columns (features per layer)
    float gamma = 1.0f;    // Scale parameter
    float beta = 0.0f;     // Bias parameter
    float epsilon = 1e-5f;  // Small value to avoid division by zero

    // Host memory allocation
    float *h_input = (float*)malloc(num_rows * num_cols * sizeof(float));
    float *h_output = (float*)malloc(num_rows * num_cols * sizeof(float));
    float *h_mean = (float*)malloc(num_rows * sizeof(float));
    float *h_variance = (float*)malloc(num_rows * sizeof(float));

    // Initialize host input data (example)
    for (int i = 0; i < num_rows * num_cols; i++) {
        h_input[i] = (float)i + 1.0f;
    }

    // Device memory allocation
    float *d_input, *d_output, *d_mean, *d_variance;
    cudaMalloc((void**)&d_input, num_rows * num_cols * sizeof(float));
    cudaMalloc((void**)&d_output, num_rows * num_cols * sizeof(float));
    cudaMalloc((void**)&d_mean, num_rows * sizeof(float));
    cudaMalloc((void**)&d_variance, num_rows * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threads_per_block = 256;  // Adjust based on your GPU
    int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    // Launch the fused kernel
    fused_layer_norm<<<num_blocks, threads_per_block>>>(d_input, d_output, d_mean, d_variance, gamma, beta, epsilon, num_rows, num_cols);

    // Copy output data from device to host
    cudaMemcpy(h_output, d_output, num_rows * num_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mean, d_mean, num_rows * sizeof(float), cudaMemcpyDeviceToHost);   // Copy mean
    cudaMemcpy(h_variance, d_variance, num_rows * sizeof(float), cudaMemcpyDeviceToHost); // Copy variance

    // Print the output
    printf("Input:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%f ", h_input[i * num_cols + j]);
        }
        printf("\n");
    }

    printf("Output:\n");
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%f ", h_output[i * num_cols + j]);
        }
        printf("\n");
    }

    printf("Mean:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f ", h_mean[i]);
    }
    printf("\n");

    printf("Variance:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f ", h_variance[i]);
    }
    printf("\n");


    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);

    // Free host memory
    free(h_input);
    free(h_output);
    free(h_mean);
    free(h_variance);

    return 0;
}
