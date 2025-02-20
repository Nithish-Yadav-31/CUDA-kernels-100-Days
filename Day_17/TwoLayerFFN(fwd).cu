#include <stdio.h>
#include <cuda_runtime.h>

// Fused Kernel for Two-Layer Forward Pass
__global__ void fused_two_layer_forward(float *input, float *weights1, float *bias1, float *weights2, float *bias2, float *output, int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        // Layer 1: Input -> Hidden
        float hidden[128]; // Local memory for hidden layer activations. Choose max hidden_size expected.
        float sum;

        for (int j = 0; j < hidden_size; j++) {
            sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                sum += input[i] * weights1[i * hidden_size + j];
            }
            hidden[j] = sum + bias1[j];  // No activation function (for simplicity)
        }

        // Layer 2: Hidden -> Output
        sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden[j] * weights2[j * output_size + idx];
        }
        output[idx] = sum + bias2[idx]; // No activation function (for simplicity)
    }
}


int main() {
    int input_size = 4;     // Number of input features
    int hidden_size = 3;    // Number of hidden neurons
    int output_size = 2;    // Number of output neurons

    // Host memory allocation
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_weights1 = (float*)malloc(input_size * hidden_size * sizeof(float));
    float *h_bias1 = (float*)malloc(hidden_size * sizeof(float));
    float *h_weights2 = (float*)malloc(hidden_size * output_size * sizeof(float));
    float *h_bias2 = (float*)malloc(output_size * sizeof(float));
    float *h_output = (float*)malloc(output_size * sizeof(float));

    // Initialize host data (example values)
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (float)i + 1.0f;
    }

    for (int i = 0; i < input_size * hidden_size; i++) {
        h_weights1[i] = (float)(i + 1.0f) * 0.1f;
    }

    for (int i = 0; i < hidden_size; i++) {
        h_bias1[i] = (float)i * 0.2f;
    }

    for (int i = 0; i < hidden_size * output_size; i++) {
        h_weights2[i] = (float)(i + 1.0f) * 0.3f;
    }

    for (int i = 0; i < output_size; i++) {
        h_bias2[i] = (float)i * 0.4f;
    }

    // Device memory allocation
    float *d_input, *d_weights1, *d_bias1, *d_weights2, *d_bias2, *d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_bias1, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_weights2, hidden_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_bias2, output_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights1, h_weights1, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1, h_bias1, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, h_weights2, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, h_bias2, output_size * sizeof(float), cudaMemcpyDeviceToHost); // Typo fix: Copy to device
    cudaMemcpy(d_output, h_output, output_size * sizeof(float), cudaMemcpyHostToDevice);  // To initialize to zero (optional, but good practice)


    // Kernel launch configuration
    int threads_per_block = 256;  // Adjust based on your GPU
    int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;

    // Launch the fused kernel
    fused_two_layer_forward<<<num_blocks, threads_per_block>>>(d_input, d_weights1, d_bias1, d_weights2, d_bias2, d_output, input_size, hidden_size, output_size);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    printf("Output:\n");
    for (int i = 0; i < output_size; i++) {
        printf("Output[%d]: %f\n", i, h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_weights1);
    free(h_bias1);
    free(h_weights2);
    free(h_bias2);
    free(h_output);

    return 0;
}
