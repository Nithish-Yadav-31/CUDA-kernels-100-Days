Vector normalization involves converting a vector to a unit vector, which has the same direction but a length (or magnitude) of 1. This is done by dividing each component of the vector by the vector's magnitude.

The formula for normalizing a vector \( \mathbf{v} = (v_1, v_2, ..., v_n) \) is:

\[
\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\| \mathbf{v} \|}
\]

Where \( \| \mathbf{v} \| \) (the magnitude) is computed as:

\[
\| \mathbf{v} \| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}
\]

### Key Steps in the Code:

1. **Input Vector Initialization**: We initialize the input vector `h_V` on the host with values (1, 2, 3,...).
   
2. **Memory Allocation**: Memory for the vector `h_V` and its normalized version `h_V_norm` is allocated on both the host and the device.
   
3. **Magnitude Calculation**: The magnitude of the vector is calculated on the host (`magnitude = sqrt(sum of squares)`).

4. **CUDA Kernel**: The `NormalizeVector` kernel normalizes each element of the vector by dividing it by the magnitude.

5. **Memory Transfer**: After computation, the result is copied back from the device to the host.

6. **Freeing Memory**: The allocated memory is freed at the end.

### Output:

For an input vector `[1, 2, 3, ..., 10]`, the program will output the normalized vector (with each element divided by the magnitude). The magnitude is calculated as:

\[
\text{magnitude} = \sqrt{1^2 + 2^2 + 3^2 + ... + 10^2}
\]

This simple CUDA program normalizes a vector using parallel processing, allowing faster computation when working with larger vectors.
