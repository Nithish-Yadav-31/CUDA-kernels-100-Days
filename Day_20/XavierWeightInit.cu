#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA kernel for Xavier (Glorot) weight initialization
__global__ void xavier_weight_initialization(float *weights, int rows, int cols, unsigned int seed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // Calculate the bound based on the number of inputs and outputs
        float range = sqrtf(6.0f / (float)(rows + cols));  // Xavier Normal Initialization bound

        // Generate a pseudo-random number between -1 and 1 using a linear congruential generator (LCG)
        unsigned int tid = row * cols + col;
        unsigned int local_seed = seed + tid;  // Unique seed for each thread
        unsigned int lcg_state = local_seed;

        // LCG parameters (can be adjusted)
        unsigned int a = 1664525;
        unsigned int c = 1013904223;

        lcg_state = a * lcg_state + c;
        float random_number = (float)lcg_state / (float)UINT_MAX; // Normalize to [0, 1]
        random_number = random_number * 2.0f - 1.0f; // Scale to [-1, 1]


        // Scale the random number by the calculated range
        weights[row * cols + col] = random_number * range;
    }
}

int main() {
    int rows = 10;   // Number of rows (input features)
    int cols = 5;    // Number of columns (output neurons)

    // Host memory allocation
    float *h_weights = (float*)malloc(rows * cols * sizeof(float));

    // Device memory allocation
    float *d_weights;
    cudaMalloc((void**)&d_weights, rows * cols * sizeof(float));

    // Set the seed for the random number generator
    unsigned int seed = 12345; // Example seed, use a more robust seed in practice

    // Kernel launch configuration
    dim3 threads_per_block(16, 16); // Adjust based on your GPU.  Should be a product <= 1024 on most GPUs.  Experiment!
    dim3 num_blocks((rows + threads_per_block.x - 1) / threads_per_block.x,
                       (cols + threads_per_block.y - 1) / threads_per_block.y);

    // Launch the Xavier weight initialization kernel
    xavier_weight_initialization<<<num_blocks, threads_per_block>>>(d_weights, rows, cols, seed);

    // Copy the initialized weights from device to host
    cudaMemcpy(h_weights, d_weights, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the initialized weights (optional)
    printf("Initialized Weights:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", h_weights[i * cols + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_weights);

    // Free host memory
    free(h_weights);

    return 0;
}
