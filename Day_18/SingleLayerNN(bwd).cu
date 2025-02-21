#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Fused Kernel for Single Layer NN Backpropagation
__global__ void fused_single_layer_backprop(
    float *input,         // Input data
    float *weights,       // Weights
    float *bias,          // Bias
    float *output,        // Output (before activation)
    float *activation,    // Output after activation
    float *target,        // Target values
    float *d_weights,      // Weight gradients
    float *d_bias,         // Bias gradients
    int input_size,       // Number of input features
    int output_size,      // Number of output neurons
    float learning_rate     // Learning rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Output neuron index

    if (idx < output_size) {
        // 1. Forward Pass (within the kernel) - This is needed to calculate the activation and the loss
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = sum + bias[idx];

        // Activation Function (Sigmoid for this example)
        activation[idx] = 1.0f / (1.0f + expf(-output[idx])); // Sigmoid Activation

        // 2. Calculate Loss (Mean Squared Error) and its derivative
        float error = target[idx] - activation[idx];
        float delta = error * activation[idx] * (1.0f - activation[idx]); // Derivative of sigmoid

        // 3. Calculate Gradients
        d_bias[idx] = delta;

        for (int i = 0; i < input_size; i++) {
            d_weights[i * output_size + idx] = input[i] * delta;
        }

        // 4. Update Weights and Bias (Stochastic Gradient Descent)  NOTE: Atomic operations needed for TRUE parallelism
        // Since all threads will attempt to access and modify the same global memory locations.
        // The lack of atomic operations here is for simplicity ONLY and WILL lead to incorrect results in a truly
        // parallel setting. DO NOT USE THIS CODE AS-IS for a truly parallel scenario.
        for (int i = 0; i < input_size; i++) {
            weights[i * output_size + idx] += learning_rate * d_weights[i * output_size + idx];
        }
        bias[idx] += learning_rate * d_bias[idx];

    }
}

int main() {
    // Hyperparameters
    int input_size = 2;
    int output_size = 1;
    float learning_rate = 0.1f;
    int num_epochs = 100;

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
    float *h_weights = (float*)malloc(input_size * output_size * sizeof(float));
    float *h_bias = (float*)malloc(output_size * sizeof(float));

    // Gradients (Host)
    float *h_d_weights = (float*)malloc(input_size * output_size * sizeof(float));
    float *h_d_bias = (float*)malloc(output_size * sizeof(float));

    // Intermediate Outputs (Host)
    float *h_output = (float*)malloc(output_size * sizeof(float));       // Before activation
    float *h_activation = (float*)malloc(output_size * sizeof(float));   // After Activation

    // Initialize weights and biases (small random values)
    h_weights[0] = 0.1f;
    h_weights[1] = 0.2f;
    h_bias[0] = 0.0f;

    // Device memory allocation
    float *d_input, *d_weights, *d_bias, *d_output, *d_activation, *d_target, *d_d_weights, *d_d_bias;

    cudaMalloc((void**)&d_input, input_size * num_samples * sizeof(float));  // Allocate for all samples
    cudaMalloc((void**)&d_weights, input_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_bias, output_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));
    cudaMalloc((void**)&d_activation, output_size * sizeof(float));
    cudaMalloc((void**)&d_target, output_size * num_samples * sizeof(float)); // Allocate for all samples
    cudaMalloc((void**)&d_d_weights, input_size * output_size * sizeof(float));
    cudaMalloc((void**)&d_d_bias, output_size * sizeof(float));


    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        for (int sample = 0; sample < num_samples; sample++) {
            // Copy data for the CURRENT sample from host to device
            cudaMemcpy(d_input, h_input + sample * input_size, input_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, h_target + sample, output_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, h_weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias, h_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

            // Kernel launch configuration
            int threads_per_block = 1;  // Important!  Only 1 thread per block since output_size == 1
            int num_blocks = 1;          //
            // Launch the fused kernel
            fused_single_layer_backprop<<<num_blocks, threads_per_block>>>(
                d_input, d_weights, d_bias, d_output, d_activation, d_target, d_d_weights, d_d_bias,
                input_size, output_size, learning_rate
            );

            // Copy results (weights and bias) back to host AFTER the kernel execution
            cudaMemcpy(h_weights, d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bias, d_bias, output_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_activation, d_activation, output_size * sizeof(float), cudaMemcpyDeviceToHost);  // Get Activation for loss calc


            // Calculate Loss (for monitoring)
            float error = h_target[sample] - h_activation[0]; // Single output neuron
            total_loss += error * error;

        }
        printf("Epoch %d, Loss: %f\n", epoch, total_loss / num_samples); // Average loss over samples
    }


    // Print learned weights and bias
    printf("Learned Weights:\n");
    for (int i = 0; i < input_size * output_size; i++) {
        printf("%f ", h_weights[i]);
    }
    printf("\n");

    printf("Learned Bias: %f\n", h_bias[0]);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_activation);
    cudaFree(d_target);
    cudaFree(d_d_weights);
    cudaFree(d_d_bias);

    // Free host memory
    free(h_weights);
    free(h_bias);
    free(h_d_weights);
    free(h_d_bias);
    free(h_output);
    free(h_activation);

    return 0;
}
