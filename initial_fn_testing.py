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