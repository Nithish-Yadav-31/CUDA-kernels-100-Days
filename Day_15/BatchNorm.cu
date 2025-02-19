#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h> // For sqrtf

// Kernel function for batch normalization
__global__ void batch_norm(float *input, float *output, float *mean, float *variance,
                         float *gamma, float *beta, int batch_size, int num_features, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index

    if (idx < num_features) {
        float x = input[blockIdx.y * num_features + idx]; //Access by batch index
        output[blockIdx.y * num_features + idx] = gamma[idx] * ((x - mean[idx]) / sqrtf(variance[idx] + epsilon)) + beta[idx];
    }
}


// Helper function to calculate mean on the host (for demonstration)
void calculate_mean_host(float *input, float *mean, int batch_size, int num_features) {
    for (int i = 0; i < num_features; i++) {
        mean[i] = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            mean[i] += input[j * num_features + i];
        }
        mean[i] /= (float)batch_size;
    }
}

// Helper function to calculate variance on the host (for demonstration)
void calculate_variance_host(float *input, float *mean, float *variance, int batch_size, int num_features) {
    for (int i = 0; i < num_features; i++) {
        variance[i] = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            variance[i] += powf(input[j * num_features + i] - mean[i], 2.0f);
        }
        variance[i] /= (float)batch_size;
    }
}


int main() {
    int batch_size = 2;    // Number of samples in the batch
    int num_features = 4;  // Number of features per sample
    float epsilon = 1e-5f; // Small constant to avoid division by zero

    // Host memory allocation
    float *h_input = (float*)malloc(batch_size * num_features * sizeof(float));
    float *h_output = (float*)malloc(batch_size * num_features * sizeof(float));
    float *h_mean = (float*)malloc(num_features * sizeof(float));
    float *h_variance = (float*)malloc(num_features * sizeof(float));
    float *h_gamma = (float*)malloc(num_features * sizeof(float));
    float *h_beta = (float*)malloc(num_features * sizeof(float));

    // Initialize host data (example values)
    for (int i = 0; i < batch_size * num_features; i++) {
        h_input[i] = (float)i + 1.0f; // Example input data
    }
    for (int i = 0; i < num_features; i++) {
        h_gamma[i] = 1.0f;      // Example gamma (scale)
        h_beta[i] = 0.0f;       // Example beta (bias)
    }

    // Calculate mean and variance on the host (for simplicity).  In a real application, you
    // would likely calculate these on the GPU as well.
    calculate_mean_host(h_input, h_mean, batch_size, num_features);
    calculate_variance_host(h_input, h_mean, h_variance, batch_size, num_features);

    // Device memory allocation
    float *d_input;
    float *d_output;
    float *d_mean;
    float *d_variance;
    float *d_gamma;
    float *d_beta;

    cudaMalloc((void**)&d_input, batch_size * num_features * sizeof(float));
    cudaMalloc((void**)&d_output, batch_size * num_features * sizeof(float));
    cudaMalloc((void**)&d_mean, num_features * sizeof(float));
    cudaMalloc((void**)&d_variance, num_features * sizeof(float));
    cudaMalloc((void**)&d_gamma, num_features * sizeof(float));
    cudaMalloc((void**)&d_beta, num_features * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, batch_size * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, num_features * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threads_per_block = 256;  // Adjust based on your GPU
    int num_blocks_features = (num_features + threads_per_block - 1) / threads_per_block;
    dim3 gridDim(num_blocks_features, batch_size);  // One block per feature, and grid per batch.
    dim3 blockDim(threads_per_block);

    // Launch the kernel
    batch_norm<<<gridDim, blockDim>>>(d_input, d_output, d_mean, d_variance, d_gamma, d_beta, batch_size, num_features, epsilon);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, batch_size * num_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    printf("Output:\n");
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_features; j++) {
            printf("Output[%d][%d]: %f\n", i, j, h_output[i * num_features + j]);
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    // Free host memory
    free(h_input);
    free(h_output);
    free(h_mean);
    free(h_variance);
    free(h_gamma);
    free(h_beta);

    return 0;
}
