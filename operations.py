import numpy as np
from typing import Tuple, Optional, Union, Literal
from numpy.lib.stride_tricks import as_strided


class Conv2D:
    def __init__(self,
                 input_channel: int,
                 output_channels: int,
                 kernel_size: Union[int, Tuple],
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

        if not isinstance(kernel_size, Tuple):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if not isinstance(stride, Tuple):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if not isinstance(padding, Tuple):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        if not isinstance(dilation, Tuple):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.use_bias = bias
        self.padding_mode = padding_mode

        k = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        limit = np.sqrt(1 / k)
        self.weights = np.random.uniform(-limit, limit, size=(self.output_channels,
                                                              self.input_channel,
                                                              self.kernel_size[0],
                                                              self.kernel_size[1]))
        if self.use_bias:
            self.bias = np.random.uniform(-limit, limit, size=(output_channels,))


    def forward(self, data: np.ndarray):
        x = data.copy()
        x_dim = x.ndim
        if x_dim == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        elif x_dim == 3:
            x = x[np.newaxis, :, :, :]

        if any(self.padding):
            pad_width = (
                (0, 0),  # no padding on batch dimension
                (0, 0),  # no padding on channel dimension
                (self.padding[0], self.padding[0]),  # height padding
                (self.padding[1], self.padding[1])  # width padding
            )
            x = np.pad(x, pad_width=pad_width, mode=self.padding_mode)
            N, in_C, in_H, in_W = x.shape
        else:
            N, in_C, in_H, in_W = x.shape

        out_H = (in_H - self.dilation[0] * (self.kernel_size[0] - 1) -
                 1) // self.stride[0] + 1
        out_W = (in_W - self.dilation[1] * (self.kernel_size[1] - 1) -
                 1) // self.stride[1] + 1

        # flatteing_kernels: [out_C, in_C, kH, kW] -> [out_C, in_C * kH * kW]
        kernels_flat = self.weights.reshape(self.output_channels, -1)

        # Create sliding windows view with as_strided
        # Output shape: (N, in_C, out_H, out_W, kH, kW)
        shape = (N, in_C, out_H, out_W, self.kernel_size[0], self.kernel_size[1])
        strides = (
            x.strides[0],                      # batch stride
            x.strides[1],                      # channel stride
            x.strides[2] * self.stride[0],     # output height stride
            x.strides[3] * self.stride[1],     # output width stride
            x.strides[2] * self.dilation[0],   # kernel height stride
            x.strides[3] * self.dilation[1]    # kernel width stride
        )
        receptive_fields = as_strided(x, shape=shape, strides=strides)
        # (N, in_C, out_H, out_W, kH, kW) -> (N*out_H*out_W, in_C*kH*kW)
        receptive_fields = np.transpose(receptive_fields, (0, 2, 3, 1, 4, 5)).reshape(N * out_H * out_W, -1)
        # (N * out_ H * out_W, in_C * kH * kW) @ (in_C * kH * kW, out_C)
        output = receptive_fields @ kernels_flat.T  # (N * out_H * out_W, out_C)
        output = output.reshape(N, out_H, out_W, self.output_channels).transpose(0, 3, 1, 2)
        output += self.bias.reshape(1, -1, 1, 1)
        return output



class MaxPool2D:
    def __init__(self,
                 kernel_size: Union[int, Tuple],
                 stride: Optional[Union[int, Tuple[int, int]]] = 1,
                 padding: Optional[Union[int, Tuple[int, int]]] = 0,
                 dilation: Optional[Union[int, Tuple[int, int]]] = 1):
        
        if not isinstance(kernel_size, Tuple):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if not isinstance(stride, Tuple):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if not isinstance(padding, Tuple):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        if not isinstance(dilation, Tuple):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation


    def forward(self, data: np.ndarray):
        x = data.copy()
        x_dim = x.ndim
        if x_dim == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        elif x_dim == 3:
            x = x[np.newaxis, :, :, :]

        if any(self.padding):
            pad_width = (
                (0, 0),  # no padding on batch dimension
                (0, 0),  # no padding on channel dimension
                (self.padding[0], self.padding[0]),  # height padding
                (self.padding[1], self.padding[1])  # width padding
            )
            x = np.pad(x, pad_width=pad_width, mode="constant", constant_values=-np.inf)
            N, in_C, in_H, in_W = x.shape
        else:
            N, in_C, in_H, in_W = x.shape

        out_H = (in_H - self.dilation[0] * (self.kernel_size[0] - 1) -1) // self.stride[0] + 1
        out_W = (in_W - self.dilation[1] * (self.kernel_size[1] - 1) -1) // self.stride[1] + 1
        
        shape = (N, in_C, out_H, out_W, self.kernel_size[0], self.kernel_size[1])
        strides = (
            x.strides[0],                      # batch stride
            x.strides[1],                      # channel stride
            x.strides[2] * self.stride[0],     # output height stride
            x.strides[3] * self.stride[1],     # output width stride
            x.strides[2] * self.dilation[0],   # kernel height stride
            x.strides[3] * self.dilation[1]    # kernel width stride
        )

        patches = as_strided(x, shape=shape, strides=strides)
        return np.max(patches, axis=(4, 5))
