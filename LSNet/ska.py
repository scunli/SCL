import torch
import torch.nn.functional as F
import math
from typing import Tuple


class MemoryEfficientSKA(torch.autograd.Function):
    @staticmethod
    # Static method, defines forward pass. Receives context object ctx, input tensor x and weight tensor w, returns output tensor
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        # Calculate kernel size. Weight tensor w has shape [batch, groups, 9, height, width], where 9 corresponds to the number of elements in a 3x3 convolution kernel, so ks = sqrt(9) = 3
        ks = int(math.sqrt(w.shape[2]))  # kernel size (3)
        # Calculate padding size. For a 3x3 kernel, pad = (3-1)//2 = 1, keeping output size unchanged
        pad = (ks - 1) // 2
        # Get number of groups, from the second dimension of weights
        groups = w.shape[1]
        # Calculate number of channels per group
        channels_per_group = channels // groups
        # Reshape weight tensor from [batch, groups, 9, height, width] to [batch, groups, 3, 3, height, width], convenient for subsequent element-wise operations
        w_reshaped = w.view(batch_size, groups, ks, ks, height, width)
        # Initialize output - use in-place operations to reduce memory
        output = torch.zeros_like(x)
        # Use chunk processing to reduce peak memory
        chunk_size = 4  # Process 4 batches at a time
        # Chunk processing loop:
        for b_start in range(0, batch_size, chunk_size): # Start index of current chunk
            b_end = min(b_start + chunk_size, batch_size) # End index of current chunk
            current_batch_size = b_end - b_start # Actual size of current chunk

            # Only process padding for current batch
            x_chunk = x[b_start:b_end]  # Extract input chunk of current batch
            x_padded = F.pad(x_chunk, (pad, pad, pad, pad), mode='constant', value=0) # Pad current chunk to handle boundary pixels

            # b: batch index within current chunk   g: group index      Calculate channel range for current group
            for b in range(current_batch_size):
                for g in range(groups):
                    start_ch = g * channels_per_group
                    end_ch = (g + 1) * channels_per_group

                    # Extract input data of current group within the current batch sample
                    group_input = x_padded[b:b + 1, start_ch:end_ch]
                    # Iterate over each spatial position of the output feature map
                    for h in range(height):
                        for w_idx in range(width):
                            # Extract local region (3x3 block) corresponding to the convolution kernel from the padded input
                            patch = group_input[:, :, h:h + ks, w_idx:w_idx + ks]

                            # Get convolution kernel weights at the current position
                            current_weights = w_reshaped[b_start + b, g, :, :, h, w_idx]

                            # Perform element-wise multiplication and summation to implement convolution:
                            # patch * current_weights.view(1, 1, ks, ks): element-wise multiply input block with weights
                            # .sum(dim=(2, 3)): sum over spatial dimensions
                            result = (patch * current_weights.view(1, 1, ks, ks)).sum(dim=(2, 3))
                            # Write the result to the corresponding position of the output tensor
                            output[b_start + b, start_ch:end_ch, h, w_idx] = result[0]
        # Save intermediate variables needed for backward pass
        ctx.save_for_backward(x, w)
        ctx.ks, ctx.pad, ctx.groups, ctx.channels_per_group = ks, pad, groups, channels_per_group

        return output

    @staticmethod
    # Define backward pass method. Receives context object ctx and output gradient grad_output, returns input gradient and weight gradient (weight gradient is None here)
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Retrieve tensors saved during forward pass from context: input x and weights w
        x, w = ctx.saved_tensors
        # Retrieve parameters saved during forward pass from context
        ks, pad, groups, channels_per_group = ctx.ks, ctx.pad, ctx.groups, ctx.channels_per_group
        # Get dimensions of input tensor
        batch_size, channels, height, width = x.shape

        # Initialize input gradient tensor with same shape as input x, memory-efficient gradient computation
        grad_x = torch.zeros_like(x)

        # Set chunk size, same memory optimization strategy as forward pass
        chunk_size = 4
        # Chunk processing loop, same chunking logic as forward pass
        for b_start in range(0, batch_size, chunk_size):
            b_end = min(b_start + chunk_size, batch_size)
            current_batch_size = b_end - b_start

            # Extract input chunk of current batch and corresponding output gradient chunk
            x_chunk = x[b_start:b_end]
            grad_output_chunk = grad_output[b_start:b_end]
            # Pad current input chunk for extracting local regions
            x_padded = F.pad(x_chunk, (pad, pad, pad, pad), mode='constant', value=0)
            # Initialize padded gradient tensor for accumulating gradient contributions
            grad_x_padded = F.pad(torch.zeros_like(x_chunk), (pad, pad, pad, pad), mode='constant', value=0)
            # Extract weight chunk of current batch and reshape to [current_batch_size, groups, 3, 3, height, width]
            w_chunk = w[b_start:b_end]
            w_reshaped = w_chunk.view(current_batch_size, groups, ks, ks, height, width)
            # Start of three-level nested loops: b: batch index within current chunk     g: group index      Calculate channel range for current group
            for b in range(current_batch_size):
                for g in range(groups):
                    start_ch = g * channels_per_group
                    end_ch = (g + 1) * channels_per_group

                    for h in range(height):  # Iterate over each spatial position of the output feature map
                        for w_idx in range(width):
                            current_weights = w_reshaped[b, g, :, :, h, w_idx] # Get convolution kernel weights at current position
                            # Extract output gradient at current position. Shape is [1, channels_per_group, 1, 1]
                            grad_patch = grad_output_chunk[b:b + 1, start_ch:end_ch, h:h + 1, w_idx:w_idx + 1]

                            # Compute gradient contribution, multiply output gradient by weights to get contribution to input gradient
                            # Broadcasting is used here: grad_patch shape [1, C, 1, 1], current_weights shape [1, 1, 3, 3]
                            # Result shape [1, C, 3, 3]
                            grad_contribution = grad_patch * current_weights.view(1, 1, ks, ks)
                            # Accumulate gradient contribution into padded gradient tensor:
                            # h:h + ks, w_idx:w_idx + ks corresponds to input region affected by the kernel (3x3 region)
                            # Use += because multiple output positions may contribute to the same input position
                            grad_x_padded[b:b + 1, start_ch:end_ch, h:h + ks, w_idx:w_idx + ks] += grad_contribution

            # Remove padding to get actual gradient of current chunk
            if pad > 0:
                grad_x_chunk = grad_x_padded[:, :, pad:-pad, pad:-pad]
            else:
                grad_x_chunk = grad_x_padded
            # Assign gradient of current chunk to the corresponding position in the final gradient tensor
            grad_x[b_start:b_end] = grad_x_chunk

        # Weight gradient set to None
        grad_w = None

        return grad_x, grad_w


class SKA(torch.nn.Module):
    """
    Memory efficient SKA module
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSKA.apply(x, w)


def memory_usage_test():
    """Test memory usage"""
    print("=" * 70)
    print("Memory Usage Test")
    print("=" * 70)

    ska = SKA()

    # Test with same dimensions as error report
    batch_size, channels, height, width = 16, 128, 88, 176
    groups = 8

    print(f"Test dimensions: [{batch_size}, {channels}, {height}, {width}]")

    # Check current GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        print(f"Initial GPU memory usage: {initial_memory:.2f} GB")

