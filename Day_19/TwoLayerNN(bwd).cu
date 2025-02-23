#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Fused Kernel for Two-Layer NN Backpropagation
__global__ void fused_two_layer_backprop(
    float *input,         // Input data
    float *weights1,      // Weights Layer 1
    float *bias1,         // Bias Layer 1
    float *weights2,      // Weights Layer 2
    float *bias2,         // Bias Layer 2
    float *hidden_output, // Hidden layer output (before activation)
    float *hidden_activation, // Hidden layer output (after activation)
    float *output,        // Output layer output (before activation)
    float *activation,    // Output layer output (after activation)
    float *target,        // Target values
    float *d_weights1,     // Weight gradients Layer 1
    float *d_bias1,        // Bias gradients Layer 1
    float *d_weights2,     // Weight gradients Layer 2
    float *d_bias2,        // Bias gradients Layer 2
    int input_size,       // Number of input features
    int hidden_size,      // Number of hidden neurons
    int output_size,      // Number of output neurons
    float learning_rate     // Learning rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Output neuron index

    if (idx < output_size) {
        // ----------------------- Forward Pass -----------------------
        // Layer 1: Input -> Hidden
        float hidden_sum[128]; // Local memory for hidden layer pre-activation
        for (int j = 0; j < hidden_size; j++) {
            hidden_sum[j] = 0.0f;
            for (int i = 0; i < input_size; i++) {
                hidden_sum[j] += input[i] * weights1[i * hidden_size + j];
            }
            hidden_output[j] = hidden_sum[j] + bias1[j];
            hidden_activation[j] = 1.0f / (1.0f + expf(-hidden_output[j]));  // Sigmoid Activation
        }

        // Layer 2: Hidden -> Output
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden_activation[j] * weights2[j * output_size + idx];
        }
        output[idx] = sum + bias2[idx];
        activation[idx] = 1.0f / (1.0f + expf(-output[idx]));  // Sigmoid Activation

        // ----------------------- Backward Pass -----------------------
        // 1. Calculate Output Layer Error and Delta
        float error = target[idx] - activation[idx];
        float delta2 = error * activation[idx] * (1.0f - activation[idx]);  // Derivative of sigmoid

        // 2. Calculate Hidden Layer Deltas
        float delta1[128];  // Local memory
        for (int j = 0; j < hidden_size; j++) {
            float sum_delta2_weights = 0.0f;
            for (int k = 0; k < output_size; k++) {
                sum_delta2_weights += delta2 * weights2[j * output_size + k]; // weight2[j,k]
            }
            delta1[j] = sum_delta2_weights * hidden_activation[j] * (1.0f - hidden_activation[j]);
        }

        // 3. Calculate Gradients and Update Weights/Biases (Layer 2)
        d_bias2[idx] = delta2;
        for (int j = 0; j < hidden_size; j++) {
            d_weights2[j * output_size + idx] = hidden_activation[j] * delta2;

            // Update weights2  (REQUIRES ATOMIC OPERATIONS FOR TRUE PARALLELISM)
            weights2[j * output_size + idx] += learning_rate * d_weights2[j * output_size + idx];
        }
        bias2[idx] += learning_rate * d_bias2[idx]; // Update bias2 (REQUIRES ATOMIC OPERATIONS FOR TRUE PARALLELISM)

        // 4. Calculate Gradients and Update Weights/Biases (Layer 1)
        for (int j = 0; j < hidden_size; j++) {
            d_bias1[j] = delta1[j];
            for (int i = 0; i < input_size; i++) {
                d_weights1[i * hidden_size + j] = input[i] * delta1[j];

                 // Update weights1 (REQUIRES ATOMIC OPERATIONS FOR TRUE PARALLELISM)
                weights1[i * hidden_size + j] += learning_rate * d_weights1[i * hidden_size + j];
            }

            bias1[j] += learning_rate * d_bias1[j];  // Update bias1 (REQUIRES ATOMIC OPERATIONS FOR TRUE PARALLELISM)
        }
    }
}


int main() {
    // Hyperparameters
    int input_size = 2;
    int hidden_size = 3;
    int output_size = 1;
    float learning_rate = 0.1f;
    int num_epochs = 1000; // Increased for better convergence.

    // Training Data (Simple XOR)
    float h_input[4 * 2] = {  // 4 samples, 2 features
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    float h_target[4] = {  // XOR targets
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };
    int num_samples = 4;  // Number of training samples

    // Model Parameters (Host)
    float *h_weights1 = (float*)malloc(input_size * hidden_size * sizeof(float));
    float *h_bias1 = (float*)malloc(hidden_size * sizeof(float));
    float *h_weights2 = (float*)malloc(hidden_size * output_size * sizeof(float));
    float *h_bias2 = (float*)malloc(output_size * sizeof(float));

    // Gradients (Host)
    float *h_d_weights1 = (float*)malloc(input_size * hidden_size * sizeof(float));
    float *h_d_bias1 = (float*)malloc(hidden_size * sizeof(float));
    float *h_d_weights2 = (float*)malloc(hidden_size * output_size * sizeof(float));
    float *h_d_bias2 = (float*)malloc(output_size * sizeof(float));

    // Intermediate Outputs (Host)
    float *h_hidden_output = (float*)malloc(hidden_size * sizeof(float));
    float *h_hidden_activation = (float*)malloc(hidden_size * sizeof(float));
    float *h_output = (float*)malloc(output_size * sizeof(float));
    float *h_activation = (float*)malloc(output_size * sizeof(float));

    // Initialize weights and biases (small random values)
    // Layer 1
    h_weights1[0] = 0.1f; h_weights1[1] = 0.2f; h_weights1[2] = 0.3f;
    h_weights1[3] = 0.4f; h_weights1[4] = 0.5f; h_weights1[5] = 0.6f;
    h_bias1[0] = 0.0f; h_bias1[1] = 0.0f; h_bias1[2] = 0.0f;

    // Layer 2
    h_weights2[0] = 0.7f; h_weights2[1] = 0.8f; h_weights2[2] = 0.9f;
    h_bias2[0] = 0.0f;

    // Device memory allocation
    float *d_input, *d_weights1, *d_bias1, *d_weights2, *d_bias2, *d_hidden_output, *d_hidden_activation, *d_output, *d_activation, *d_target, *d_d_weights1, *d_d_bias1, *d_d_weights2, *d_d_bias2;

    cudaMalloc((void**)&d_input, input_size * num_samples * sizeof(float)); // Allocate for all samples
    cudaMalloc((void**)&d_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_bias1, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_weights2, hidden_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_bias2, output_size * sizeof(float));
    cudaMalloc((void**)&d_hidden_output, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_hidden_activation, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));
    cudaMalloc((void**)&d_activation, output_size * sizeof(float));
    cudaMalloc((void**)&d_target, output_size * num_samples * sizeof(float)); // Allocate for all samples
    cudaMalloc((void**)&d_d_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc((void**)&d_d_bias1, hidden_size * sizeof(float));
    cudaMalloc((void**)&d_d_weights2, hidden_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_d_bias2, output_size * sizeof(float));


    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        for (int sample = 0; sample < num_samples; sample++) {
            // Copy data for the CURRENT sample from host to device
            cudaMemcpy(d_input, h_input + sample * input_size, input_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, h_target + sample, output_size * sizeof(float), cudaMemcpyHostToDevice);

            cudaMemcpy(d_weights1, h_weights1, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias1, h_bias1, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights2, h_weights2, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias2, h_bias2, output_size * sizeof(float), cudaMemcpyHostToDevice);



            // Kernel launch configuration (One thread per output neuron)
            int threads_per_block = 1;
            int num_blocks = 1;  // One block, one thread for the single output neuron.

            // Launch the fused kernel
            fused_two_layer_backprop<<<num_blocks, threads_per_block>>>(
                d_input, d_weights1, d_bias1, d_weights2, d_bias2, d_hidden_output, d_hidden_activation, d_output, d_activation, d_target,
                d_d_weights1, d_d_bias1, d_d_weights2, d_d_bias2, input_size, hidden_size, output_size, learning_rate
            );

            // Copy results back to host (weights and biases)
            cudaMemcpy(h_weights1, d_weights1, input_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bias1, d_bias1, hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_weights2, d_weights2, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bias2, d_bias2, output_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_activation, d_activation, output_size*sizeof(float), cudaMemcpyDeviceToHost);

            // Calculate Loss (for monitoring)
            float error = h_target[sample] - h_activation[0];
            total_loss += error * error;
        }
        printf("Epoch %d, Loss: %f\n", epoch, total_loss / num_samples);
    }

    // Print learned weights and biases
    printf("Learned Weights Layer 1:\n");
    for (int i = 0; i < input_size * hidden_size; i++) {
        printf("%f ", h_weights1[i]);
    }
    printf("\n");

    printf("Learned Biases Layer 1:\n");
    for (int i = 0; i < hidden_size; i++) {
        printf("%f ", h_bias1[i]);
    }
    printf("\n");

    printf("Learned Weights Layer 2:\n");
    for (int i = 0; i < hidden_size * output_size; i++) {
        printf("%f ", h_weights2[i]);
    }
    printf("\n");

    printf("Learned Bias Layer 2: %f\n", h_bias2[0]);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);
    cudaFree(d_hidden_output);
    cudaFree(d_hidden_activation);
    cudaFree(d_output);
    cudaFree(d_activation);
    cudaFree(d_target);
    cudaFree(d_d_weights1);
    cudaFree(d_d_bias1);
    cudaFree(d_d_weights2);
    cudaFree(d_d_bias2);

    // Free host memory
    free(h_weights1);
    free(h_bias1);
    free(h_weights2);
    free(h_bias2);
    free(h_hidden_output);
    free(h_hidden_activation);
    free(h_output);
    free(h_activation);

    return 0;
}
