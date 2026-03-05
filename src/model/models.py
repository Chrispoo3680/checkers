import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = F.relu(x)

        return x


class CheckersNetV1(nn.Module):
    def __init__(
        self, input_planes, policy_planes, channels=64, num_blocks=8, temperature=1.0
    ):
        super().__init__()

        self.temperature = temperature

        # Initial convolution
        self.conv_input = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual trunk
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # ---------------- POLICY HEAD ----------------
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_planes)

        # ---------------- VALUE HEAD ----------------
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):

        # Initial conv
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)

        # Residual trunk
        for block in self.res_blocks:
            x = block(x)

        # ================= POLICY =================
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)

        p = torch.flatten(p, start_dim=1)
        p = self.policy_fc(p)  # logits (no softmax here)

        # ================= VALUE =================
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)

        v = torch.flatten(v, start_dim=1)
        v = self.value_fc1(v)
        v = F.relu(v)

        v = self.value_fc2(v)
        v = torch.tanh(v)  # bounded evaluation

        return p / self.temperature, v
