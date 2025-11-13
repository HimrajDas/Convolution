import numpy as np
from typing import Tuple, Optional, Union, Literal
from numpy.lib.stride_tricks import as_strided


# TODO: Use loops first, then use vectorization. [Fucking loop is slow as fuck.]

class Conv2D:
    def __init__(self,
                 input_channel: int,
                 output_channels: int,
                 kernel_size: int | Tuple,
                 stride: Optional[Union[int, Tuple[int, int]]] = 1,
                 padding: Optional[Union[int, Tuple[int, int]]] = 0,
                 dilation: Optional[Union[int, Tuple[int, int]]] = 1,
                 bias: Optional[bool] = True,
                 padding_mode: Optional[Literal[
                     "constant", "reflect", "edge", "maximum",
                     "symmetric", "linear_ramp"]] = "constant",
                 ):
        self.input_channel = input_channel
        self.output_channels = output_channels
        assert self.input_channel >= 2, "Invalid input number of channels!!"

        if not isinstance(kernel_size, Tuple):
            self.kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, Tuple):
            self.stride = (stride, stride)
        if not isinstance(padding, Tuple):
            self.padding = (padding, padding)
        if not isinstance(dilation, Tuple):
            self.dilation = (dilation, dilation)

        self.bias = bias
        self.padding_mode = padding_mode


    def calculate_output_shape(self, matrix_height: int, matrix_width: int):
        """Returns the skeletons of the output matrix."""
        out_H = (matrix_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) -
                 1) // self.stride[0] + 1
        out_W = (matrix_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) -
                 1) // self.stride[1] + 1
        return np.zeros((self.output_channels, out_H, out_W), np.float32), out_H, out_W


    def weight_initialization(self):
        """weight initialization technique."""
        ...


    def initialize_kernels_and_bias(self):
        """Initializes the kernels and biases ♋ """
        # kernel size: (o_ch, in_ch, kernel_height, kernel_width)
        k = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        kernels = np.random.uniform(-np.sqrt(1 / k),
                                    np.sqrt(1 / k),
                                    size=(self.output_channels, self.input_channel,
                                          self.kernel_size[0], self.kernel_size[1]))
        bias = np.random.uniform(-np.sqrt(1 / k), np.sqrt(1 / k), size=(self.output_channels,))
        return kernels, bias


    def apply_conv2d(self, matrices: np.ndarray):
        """Applies a conv2D operation on top of an input image or batch (fully vectorized and optimized)."""

        # Handle different input dimensions
        original_ndim = matrices.ndim

        if original_ndim == 2:
            # Single channel, single image: (H, W) -> (1, 1, H, W)
            matrices = matrices[np.newaxis, np.newaxis, :, :]
        elif original_ndim == 3:
            # Single image with channels: (C, H, W) -> (1, C, H, W)
            matrices = matrices[np.newaxis, :, :, :]
        # else: already batched (N, C, H, W)

        N, in_C, in_H, in_W = matrices.shape

        # Apply padding if needed
        if any(self.padding):
            pad_width = (
                (0, 0),  # no padding on batch dimension
                (0, 0),  # no padding on channel dimension
                (self.padding[0], self.padding[0]),  # height padding
                (self.padding[1], self.padding[1])  # width padding
            )
            matrices = np.pad(matrices, pad_width=pad_width, mode=self.padding_mode)
            _, _, in_H, in_W = matrices.shape

        # Calculate output shape
        output_matrix, out_H, out_W = self.calculate_output_shape(in_H, in_W)
        kernels, biases = self.initialize_kernels_and_bias()

        # Pre-flatten kernels: (out_C, in_C, kH, kW) -> (out_C, in_C*kH*kW)
        kernels_flat = kernels.reshape(self.output_channels, -1)

        # Create sliding windows view with as_strided
        # Output shape: (N, in_C, out_H, out_W, kH, kW)
        shape = (N, in_C, out_H, out_W, self.kernel_size[0], self.kernel_size[1])
        strides = (
            matrices.strides[0],  # batch stride
            matrices.strides[1],  # channel stride
            matrices.strides[2] * self.stride[0],  # output height stride
            matrices.strides[3] * self.stride[1],  # output width stride
            matrices.strides[2] * self.dilation[0],  # kernel height stride
            matrices.strides[3] * self.dilation[1]  # kernel width stride
        )
        receptive_fields = as_strided(matrices, shape=shape, strides=strides)
        # Reshape for efficient matrix multiplication
        # From: (N, in_C, out_H, out_W, kH, kW)
        # To:   (N*out_H*out_W, in_C*kH*kW)
        receptive_fields = receptive_fields.transpose(0, 2, 3, 1, 4, 5).reshape(N * out_H * out_W, -1)

        # Perform convolution via GEMM
        # (N*out_H*out_W, in_C*kH*kW) @ (in_C*kH*kW, out_C) -> (N*out_H*out_W, out_C)
        output_flat = receptive_fields @ kernels_flat.T

        # Reshape to final output shape
        # From: (N*out_H*out_W, out_C) 
        # To:   (N, out_C, out_H, out_W)
        output_matrix = output_flat.reshape(N, out_H, out_W, self.output_channels).transpose(0, 3, 1, 2)
        output_matrix += biases.reshape(1, -1, 1, 1)    # Add biases
        return output_matrix
