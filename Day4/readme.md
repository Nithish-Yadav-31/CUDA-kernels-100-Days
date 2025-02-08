### Explanation of Key Parts:

1. **Matrix and Vector Initialization with `float`:**
   - `h_A` is a matrix of `float` values of size `N x N`.
   - `h_B` is a vector of `float` values of size `N`.
   - `h_C` is a vector of `float` values of size `N` (to store the result of the matrix-vector multiplication).

2. **CUDA Kernel (`MatrixVecMul`):**
   - This kernel calculates each element of the result vector `C` by performing a dot product of each row of matrix `A` and vector `B`.
   - `sum += A[i * N + j] * B[j];` computes the dot product between row `i` of matrix `A` and vector `B` (all using `float`).

3. **Grid and Block Dimensions:**
   - The block size is `16` threads per block. You can change this based on your GPU's capabilities.
   - `gridDim` is calculated to ensure that we have enough blocks to process all the rows of the matrix.

4. **Memory Allocation:**
   - Memory for the matrix `A` and vectors `B` and `C` is allocated both on the host (`malloc`) and device (`cudaMalloc`).
   - `cudaMemcpy` is used to transfer data between host and device.

5. **Result Printing:**
   - After the kernel finishes execution, the result vector `C` is copied back to the host and printed.
