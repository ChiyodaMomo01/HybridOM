import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list[0], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# --- ConvLSTM wrapper (single-step interface) ---
class ConvLSTM_NS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_first=True):
        """
        Args:
            input_dim: input channels (in_C)
            hidden_dim: hidden channels per layer (e.g., [64, 64])
            output_dim: output channels (out_C)
            num_layers: number of ConvLSTM layers
            batch_first: dimension order
        """
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=batch_first,
            return_all_layers=False
        )
        # Final 1x1 conv maps hidden channels to output channels.
        self.conv_out = nn.Conv2d(hidden_dim[-1], output_dim, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, C, H, W)
        
        # Add a dummy time dimension: (B, 1, C, H, W)
        x = x.unsqueeze(1)
        
        # ConvLSTM output: (B, T=1, hidden_dim, H, W)
        layer_output, _ = self.convlstm(x)
        
        B, T, C_hidden, H, W = layer_output.shape
        
        # Flatten time into batch for Conv2d.
        combined = layer_output.reshape(B * T, C_hidden, H, W)
        
        # Output conv: (B*T, out_C, H, W)
        output = self.conv_out(combined)
        
        # Restore time dimension: (B, T=1, out_C, H, W)
        output = output.reshape(B, T, -1, H, W)
        
        # Drop dummy time dimension: (B, out_C, H, W)
        output = output.squeeze(1)
        
        return output

if __name__ == '__main__':
    # Minimal sanity check.
    batch_size = 1
    in_channels = 99
    out_channels = 93
    height = 360
    width = 720

    # Synthetic input (B, in_C, H, W)
    x = torch.randn((batch_size, in_channels, height, width))
    print(f"Input shape: {x.shape}")

    # Initialize model
    model = ConvLSTM_NS(
        input_dim=in_channels, 
        hidden_dim=[64, 64],
        output_dim=out_channels, 
        num_layers=2, 
        batch_first=True
    )

    # Forward-only check
    with torch.no_grad():
        pred = model(x)

    print(f"Output shape: {pred.shape}")
    
    expected_shape = (batch_size, out_channels, height, width)
    assert pred.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {pred.shape}"
    print("Sanity check passed.")