import numpy as np
from typing import Tuple, Optional, Union, Literal


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

    def apply_conv2d(self, input_matrix: np.ndarray):
        """Applies a conv2D operation on top of an input image."""
        input_matrix = input_matrix.copy()

        if any(self.padding):
            if self.input_channel < 3:
                pad_width = ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]))
            else:
                pad_width = ((0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]))
            input_matrix = np.pad(input_matrix, pad_width=pad_width, mode=self.padding_mode)

        _, in_H, in_W = input_matrix.shape
        output_matrix, out_H, out_W = self.calculate_output_shape(in_H, in_W)
        kernels, biases = self.initialize_kernels_and_bias()
        for c in range(self.output_channels):
            current_kernel = kernels[c]
            for h in range(out_H):
                for w in range(out_W):
                    h_start, w_start = h * self.stride[0], w * self.stride[1]
                    if input_matrix.ndim == 2:
                        receptive_field = input_matrix[
                                          h_start: (h_start + self.kernel_size[0]) * self.dilation[0]: self.dilation[0],
                                          w_start: (w_start + self.kernel_size[1]) * self.dilation[1]: self.dilation[1]]
                    else:
                        receptive_field = input_matrix[:,
                                          h_start: (h_start + self.kernel_size[0]) * self.dilation[0]: self.dilation[0],
                                          w_start: (w_start + self.kernel_size[1]) * self.dilation[1]: self.dilation[1]]
                    output_matrix[c, h, w] = np.sum(receptive_field * current_kernel)
            output_matrix[c, :, :] += biases[c]

        return output_matrix

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
