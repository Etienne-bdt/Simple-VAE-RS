import math

import torch
import torch.nn as nn


def calculate_padding(in_size, out_size, kernel_size, stride=1, dilation=1):
    """
    Calculate the padding needed for a convolutional layer given the kernel size and dilation.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding = max(0, (in_size - out_size * stride + effective_kernel_size - 1) // 2)
    return padding


def calculate_output_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate the output size of a convolutional layer given the input size, kernel size, stride, padding, and dilation.
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    out_size = (in_size + 2 * padding - effective_kernel_size) // stride + 1
    return out_size


class downsample_sequence(nn.Module):
    def __init__(self, in_shape, out_flattened_size, out_channels=None, num_steps=None):
        """
        Args:
            in_shape: (C, H, W) tuple for input
            out_flattened_size: int, desired flattened output size (must be divisible by out_channels and form a square shape)
            out_channels: output channels (optional, will be auto-calculated if not given)
            num_steps: number of downsampling steps (optional, will be inferred if not given)
        """
        super(downsample_sequence, self).__init__()
        assert out_flattened_size is not None, "Must specify out_flattened_size"
        # Auto-calculate out_channels and H=W if not provided
        if out_channels is None:
            for hw in range(int(math.sqrt(out_flattened_size)), 0, -1):
                if out_flattened_size % (hw * hw) == 0:
                    out_channels = out_flattened_size // (hw * hw)
                    target_h = target_w = hw
                    break
            else:
                raise ValueError(
                    "Cannot find suitable (out_channels, H, W) for given flattened size"
                )
        else:
            total_spatial = out_flattened_size // out_channels
            target_h = target_w = int(total_spatial**0.5)
            assert target_h * target_w * out_channels == out_flattened_size, (
                "out_flattened_size must be divisible by out_channels and form a square shape"
            )
        self.out_channels = out_channels
        self.target_h = target_h
        self.target_w = target_w
        self.layers = nn.ModuleList()
        c, h, w = in_shape[0], in_shape[1], in_shape[2]
        steps = num_steps or 0
        # If num_steps not given, compute minimal steps to reach target spatial size
        if num_steps is None:
            tmp_h, tmp_w = h, w
            while tmp_h > target_h and tmp_w > target_w:
                tmp_h = (tmp_h + 1) // 2
                tmp_w = (tmp_w + 1) // 2
                steps += 1
        else:
            steps = num_steps

        if steps > 0:
            max_stride_steps = 0
            tmp_h, tmp_w = h, w
            for _ in range(steps):
                if tmp_h > target_h and tmp_w > target_w:
                    tmp_h = (tmp_h + 1) // 2
                    tmp_w = (tmp_w + 1) // 2
                    max_stride_steps += 1
                else:
                    break
            # Use stride=2 for max_stride_steps, then stride=1 for the rest
            stride_plan = [2] * max_stride_steps + [1] * (steps - max_stride_steps)
        else:
            stride_plan = []
        # Plan progressive channel increase using powers of four
        if steps > 1:
            ch_progression = [
                min(out_channels, in_shape[0] * (4**i)) for i in range(steps)
            ]
            ch_progression[-1] = out_channels  # Ensure last is exactly out_channels
        else:
            ch_progression = [out_channels]
        for i in range(steps):
            is_last = i == steps - 1
            stride = stride_plan[i] if i < len(stride_plan) else 1
            kernel_size = 5 if stride == 2 else 3
            next_ch = ch_progression[i]
            out_ch = next_ch
            # For last layer, match target_h/target_w
            next_h = target_h if is_last else (h + stride - 1) // stride
            padding = calculate_padding(h, next_h, kernel_size, stride)
            self.layers.append(
                nn.Conv2d(c, out_ch, kernel_size, stride=stride, padding=padding)
            )
            if not is_last:
                self.layers.append(nn.ReLU(inplace=True))
            c = out_ch
            h = calculate_output_size(h, kernel_size, stride, padding)
            w = calculate_output_size(w, kernel_size, stride, padding)
        self.final_shape = (c, h, w)
        self.flatten = nn.Flatten()
        assert c * h * w == out_flattened_size, (
            f"Final output shape {c}x{h}x{w} does not match requested flattened size {out_flattened_size}"
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        return x


class upsample_sequence(nn.Module):
    def __init__(self, in_flattened_size, out_shape, in_channels, num_steps=None):
        """
        Args:
            in_flattened_size: int, flattened input size (must be divisible by in_channels and form a square shape)
            out_shape: (C, H, W) tuple for output
            in_channels: number of channels to unflatten to (should match last out_channels of downsample)
            num_steps: number of upsampling steps (optional, will be inferred if not given)
        """
        super(upsample_sequence, self).__init__()
        assert in_flattened_size is not None, "Must specify in_flattened_size"
        assert out_shape is not None, "Must specify out_shape (C, H, W)"
        assert in_channels is not None, "Must specify in_channels (should match downsample out_channels)"
        out_channels = out_shape[0]  # Always infer from out_shape
        self.sigmoid = nn.Sigmoid()
        target_h, target_w = out_shape[1], out_shape[2]
        self.out_channels = out_channels
        self.target_h = target_h
        self.target_w = target_w
        self.layers = nn.ModuleList()
        # Infer in_h, in_w from in_flattened_size and in_channels


        total_spatial = in_flattened_size // in_channels
        in_h = in_w = int(total_spatial ** 0.5)
        assert in_h * in_w * in_channels == in_flattened_size, "in_flattened_size must be divisible by in_channels and form a square shape"
        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w
        c, h, w = in_channels, in_h, in_w
        # If num_steps not given, compute minimal steps to reach target spatial size
        steps = num_steps or 0
        if num_steps is None:
            tmp_h, tmp_w = h, w
            while tmp_h < target_h and tmp_w < target_w:
                tmp_h = tmp_h * 2
                tmp_w = tmp_w * 2
                steps += 1
        else:
            steps = num_steps
        if steps > 0:
            max_stride_steps = 0
            tmp_h, tmp_w = h, w
            for _ in range(steps):
                if tmp_h < target_h and tmp_w < target_w:
                    tmp_h = tmp_h * 2
                    tmp_w = tmp_w * 2
                    max_stride_steps += 1
                else:
                    break
            stride_plan = [2] * max_stride_steps + [1] * (steps - max_stride_steps)
        else:
            stride_plan = []
        if steps > 1:
            ch_progression = [max(out_channels, c // (4 ** i)) for i in range(steps)]
            ch_progression[-1] = out_channels  # Ensure last is exactly out_channels
        else:
            ch_progression = [out_channels]
        self.unflatten = nn.Unflatten(1, (in_channels, in_h, in_w))
        # Improved output shape calculation for ConvTranspose2d (no padding)
        # Use kernel_size=4, stride=2, padding=1 for upsampling (standard for exact doubling)
        for i in range(steps):
            is_last = i == steps - 1
            stride = stride_plan[i] if i < len(stride_plan) else 1
            if stride == 2:
                kernel_size = 4
                padding = 1
            else:
                kernel_size = 3
                padding = 1
            next_ch = ch_progression[i]
            out_ch = next_ch
            # output_padding must be int, not tuple
            output_padding = 0
            self.layers.append(
                nn.ConvTranspose2d(c, out_ch, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
            )
            if not is_last:
                self.layers.append(nn.ReLU(inplace=True))
            # Update h, w using the ConvTranspose2d formula
            h = (h - 1) * stride - 2 * padding + kernel_size + output_padding
            w = (w - 1) * stride - 2 * padding + kernel_size + output_padding
            c = out_ch
        self.final_shape = (c, h, w)
        if not (c == out_channels and h == target_h and w == target_w):
            raise RuntimeError(f"Upsample sequence produced invalid shape {c}x{h}x{w}, expected {out_channels}x{target_h}x{target_w}. Check your configuration.")

    def forward(self, x):
        x = self.unflatten(x)
        for layer in self.layers:
            x = layer(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    # Example usage
    in_shape = (4, 32, 32)  # C, H, W
    out_flattened_size = 4096
    model = downsample_sequence(
        in_shape, out_flattened_size, num_steps=5, out_channels=256
    )
    print(model)

    # Test with a random input tensor
    x = torch.randn(1, *in_shape)  # Batch size of 1
    output = model(x)
    print("Output shape:", output.shape)

    # Example usage for upsample_sequence
    upmodel = upsample_sequence(
        in_channels=256,
        in_flattened_size=out_flattened_size, 
        out_shape=(4, 32, 32), num_steps=5
    )
    print(upmodel)

    # Test upsample with random tensor
    x = torch.randn(1, 4096)  # Batch size of 1, matching downsample output
    upoutput = upmodel(x)
    print("Upsample output shape:", upoutput.shape)
