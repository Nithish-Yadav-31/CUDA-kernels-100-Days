import torch
import time

# Define the size of the input
N = 5

# Define the input tensor
h_input = torch.arange(N, dtype=torch.float32) - N / 2.0
h_input = h_input / N

# Move the input tensor to the GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")
h_input = h_input.to(device)


# Perform the softmax operation
def softmax(x):
    # Calculate the exponential of each element
    e_x = torch.exp(x - torch.max(x))  # Subtract max for numerical stability

    # Calculate the sum of the exponentials
    sum_e_x = torch.sum(e_x)

    # Calculate the softmax
    return e_x / sum_e_x

# Measure the execution time
start_time = time.time()
h_output = softmax(h_input)
end_time = time.time()

# Print the results
for i in range(N):
    print(f"Softmax({h_input[i].item()}) = {h_output[i].item()}")

# Print the execution time
execution_time = (end_time - start_time) * 1000
print(f"Total execution time: {execution_time:.4f} ms")
