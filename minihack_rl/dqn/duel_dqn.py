import torch
import torch.nn as nn


class CustomConv2d(torch.Module):
    def __init__(self, input, out_channels, kernel_size, stride, padding=0, dilation=1, device=None):
        C_in, H_in, W_in = input

        self.H_out = (H_in + 2*padding - dilation *
                      (kernel_size - 1) - 1) / stride
        self.W_out = (W_in + 2*padding - dilation *
                      (kernel_size - 1) - 1) / stride

        self.model = (nn.Sequential(
            nn.Conv2d(C_in, out_channels, kernel_size, stride, device=device),
            nn.ReLU()
        ))

    def forward(self, x):
        return self.model(x)


class DuelDQN(torch.Module):
    def __init__(self, input_dim, output_dim, device):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = device

        self.conv1 = CustomConv2d(self.input_dim, 32, 8, 4)
        out1_dim = (self.conv1.H_out, self.conv1.W_out)

        self.conv2 = CustomConv2d(out1_dim, 64, 4, 2)
        out2_dim = (self.conv2.H_out, self.conv2.W_out)

        self.conv3 = CustomConv2d(out2_dim, 64, 3, 1)
        out3_dim = (self.conv3.H_out, self.conv3.W_out)

        self.value = nn.Sequential(
            nn.Linear(out3_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(out3_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.size(0), -1)

        V = self.value(x)

        A = self.advantage(x)

        Q_values = V + (A - A.mean())

        return Q_values
