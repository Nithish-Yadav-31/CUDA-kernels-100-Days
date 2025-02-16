#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

// Define the kernel
__global__ void maxPoolingKernel(
    float *input,         // Input feature map (global memory)
    float *output,        // Output pooled feature map (global memory)
    int inputHeight,      // Height of the input feature map
    int inputWidth,       // Width of the input feature map
    int inputChannels,    // Number of channels in the input
    int outputHeight,     // Height of the output feature map
    int outputWidth,      // Width of the output feature map
    int poolHeight,       // Height of the pooling window
    int poolWidth,        // Width of the pooling window
    int strideHeight,      // Stride in the height dimension
    int strideWidth         // Stride in the width dimension
) {
  // Get thread and block indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Calculate the output coordinates for this thread
  int outputX = bx * blockDim.x + tx; // X coordinate in output feature map
  int outputY = by * blockDim.y + ty; // Y coordinate in output feature map

  //Check the coordinates for out-of-bounds
  if (outputX >= outputWidth || outputY >= outputHeight)
  {
    return;
  }


  // Calculate the corresponding input region's starting coordinates
  int startX = outputX * strideWidth;  // Starting X coordinate in input
  int startY = outputY * strideHeight; // Starting Y coordinate in input


  // Perform max pooling within the window
  for (int channel = 0; channel < inputChannels; ++channel) {
    float maxVal = -999999.0f; // Initialize with a very small value, considering input might be negative
    for (int i = 0; i < poolHeight; ++i) {
      for (int j = 0; j < poolWidth; ++j) {
        int inputX = startX + j; // Current X coordinate within pooling window in input
        int inputY = startY + i; // Current Y coordinate within pooling window in input

        // Bounds checking: handle cases where the pool extends beyond the input
        if (inputX < inputWidth && inputY < inputHeight) {
          // Calculate the linear index into the input array
          int inputIndex = channel * inputHeight * inputWidth + inputY * inputWidth + inputX;
          float val = input[inputIndex]; // Get the value from the input
          maxVal = fmaxf(maxVal, val);    // Update maxVal if needed (using fast CUDA fmaxf)
        }
      }
    }

    // Calculate the linear index into the output array and write the max value
    int outputIndex = channel * outputHeight * outputWidth + outputY * outputWidth + outputX;
    output[outputIndex] = maxVal;
  }
}

// Example of how to launch the kernel from the host
int main() {
  // Define dimensions and parameters (example values)
  int inputHeight = 64;
  int inputWidth = 64;
  int inputChannels = 3;
  int poolHeight = 2;
  int poolWidth = 2;
  int strideHeight = 2;
  int strideWidth = 2;

  int outputHeight = (inputHeight - poolHeight) / strideHeight + 1;  // Calculate output dimensions
  int outputWidth = (inputWidth - poolWidth) / strideWidth + 1;

  // Allocate host memory (CPU memory) for input and output
  float *h_input = new float[inputHeight * inputWidth * inputChannels];
  float *h_output = new float[outputHeight * outputWidth * inputChannels];

  // Initialize input data (you would typically load this from your data)
  // Example: Fill with some values
  for (int i = 0; i < inputHeight * inputWidth * inputChannels; ++i) {
    h_input[i] = static_cast<float>(i % 256);  // Example initialization
  }

  // Allocate device memory (GPU memory)
  float *d_input;
  float *d_output;
  cudaMalloc(&d_input, inputHeight * inputWidth * inputChannels * sizeof(float));
  cudaMalloc(&d_output, outputHeight * outputWidth * inputChannels * sizeof(float));

  // Copy input data from host to device
  cudaMemcpy(d_input, h_input, inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice);

  // Define the block and grid dimensions for the kernel launch. IMPORTANT:  Choose appropriately for your GPU architecture.
  dim3 blockDim(16, 16); // Adjust block size based on your problem size and hardware. 16x16 or 32x32 are good starting points.
  dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, (outputHeight + blockDim.y - 1) / blockDim.y);

  // Launch the kernel
  maxPoolingKernel<<<gridDim, blockDim>>>(
      d_input, d_output, inputHeight, inputWidth, inputChannels,
      outputHeight, outputWidth, poolHeight, poolWidth, strideHeight, strideWidth);

  // Copy the result back from device to host
  cudaMemcpy(h_output, d_output, outputHeight * outputWidth * inputChannels * sizeof(float), cudaMemcpyDeviceToHost);

  // You can now work with the output data (h_output)

  //Print a few output values
  std::cout << "Output values:" << std::endl;
    for(int i=0; i< 10; ++i){
      std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;



  // Free allocated memory (important!)
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
