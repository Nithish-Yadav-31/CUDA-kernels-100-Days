#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void batchNormInferenceKernel(
    const float *input,     // Input tensor (N, C, H, W)
    float *output,           // Output tensor (N, C, H, W)
    const float *mean,       // Pre-calculated mean for each channel (C)
    const float *variance,   // Pre-calculated variance for each channel (C)
    const float *gamma,      // Scale parameter for each channel (C)
    const float *beta,       // Shift parameter for each channel (C)
    int N,                   // Batch size
    int C,                   // Number of channels
    int H,                   // Height
    int W,                   // Width
    float epsilon             // Small constant for numerical stability
) {
  // Get thread and block indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Calculate 1D index within the grid (maps the output space, we are operating on (N, C, H, W) tensors)
  int index = bx * blockDim.x + tx; // corresponds to `N`
  int index_by = by * blockDim.y + ty; // corresponds to C, H and W axis and so used for nested forloop processing

  // Bounds checking: Ensure threads operate within the output range
  if (index >= N || index_by >= C * H * W) { // use index as condition bound. Also very essential so each kernel invocation is guaranteed from reading from bad device pointer memory addresses!
      return;  // IMPORTANT: Exit if the thread is out of bounds
  }

  // Extract N index
  const int n = index;
  int index_h = index_by;
  int channel_stride = H * W;
  int c = index_h / channel_stride;  // Extract the channel from linear output by linear channel access, C dimension can be parallel.
  int index_h2 = index_h % channel_stride;
  int h = index_h2 / W;  //  extract Height using same access index of "remainder % H"
  int w = index_h2 % W;    // extract the remainder output by single remainder access

  // Get linear index of N*C*H*W

  int linear_idx = (n * C + c) * H * W + h * W + w;

  // Perform batch normalization for this element

  float inputValue = input[linear_idx];
  float meanValue = mean[c];
  float varianceValue = variance[c];
  float gammaValue = gamma[c];
  float betaValue = beta[c];


  // Check zero varience so the sqrt(varience + epsilon) is stable
  float stdDev = sqrtf(varianceValue + epsilon); //Using std C libraries
  //if is close enough.  Don't make stdDev is 0, causes NaN outputs! Use "if (stdDev <= someTolerantValFloatValue)"
  float normalizedValue = (inputValue - meanValue) / stdDev; // Main math operation, normalizations over device global arrays for all outputs
  float outputValue = gammaValue * normalizedValue + betaValue; // Apply shift value from global arrays

  output[linear_idx] = outputValue; // Outputs results with processed linear results;
}

// Example of how to launch the kernel from the host
int main() {
    // Define dimensions (example)
    int N = 4;       // Batch size
    int C = 3;       // Channels
    int H = 32;      // Height
    int W = 32;      // Width
    float epsilon = 1e-5f; // for numeric stabilities from preventing sqrt division by zero

    // Allocate host memory and fill data:

    //Example
    float *h_input = new float[N * C * H * W];
    float *h_output = new float[N * C * H * W];
    float *h_mean = new float[C];
    float *h_variance = new float[C];
    float *h_gamma = new float[C];
    float *h_beta = new float[C];
    // fill each of above parameters.  For now with demo numbers
    for(int n=0; n < N*C*H*W ; n++)
        h_input[n] = 1.0f * n;

    for(int c =0; c< C; c++)
    {
        h_mean[c] = 0.0f;
        h_variance[c] = 1.0f;
        h_gamma[c] = 1.0f;
        h_beta[c] = 0.0f;
    }
    // Device Allocation and data copying... (same pattern for input and outputs!)
    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_mean = nullptr;
    float *d_variance = nullptr;
    float *d_gamma = nullptr;
    float *d_beta = nullptr;
    // Use try catch (important best programming practices!!!!  Use if out of ram.)  Out-of-memeory crashing sucks badly in middle deep learning code.

     cudaMalloc(&d_input, N * C * H * W * sizeof(float));
     cudaMalloc(&d_output, N * C * H * W * sizeof(float));

     cudaMalloc(&d_mean, C * sizeof(float));
     cudaMalloc(&d_variance, C * sizeof(float));
     cudaMalloc(&d_gamma, C * sizeof(float));
     cudaMalloc(&d_beta, C * sizeof(float));

    cudaMemcpy(d_input, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32,32); // thread block parameter
    dim3 gridDim(N / blockDim.x+ 1, (C*H*W)/blockDim.y + 1);  //N/(BLOCK.X)   to be close

    // Launch kernel
    batchNormInferenceKernel<<<gridDim, blockDim>>>(
        d_input, d_output, d_mean, d_variance, d_gamma, d_beta, N, C, H, W, epsilon); // params and numbers.

    // device array pointer names, device dimensions number parameters, number

    // Copy results from device back to host
    cudaMemcpy(h_output, d_output, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // Use `h_output`. Example:

    for(int i=0;i < 10; i++){
        std::cout << " output values with number" <<  h_output[i]<< "index[" <<i << "] = " << h_output[i]<< std::endl;
    }

    // Memory deallocations (use try catch out of bound error here for ram!)

    delete[] h_input;
    delete[] h_output;
    delete[] h_mean;
    delete[] h_variance;
    delete[] h_gamma;
    delete[] h_beta;


     cudaFree(d_input);
     cudaFree(d_output);
     cudaFree(d_mean);
     cudaFree(d_variance);
     cudaFree(d_gamma);
     cudaFree(d_beta);



    return 0;
}
