import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------- Residual Block -------------------
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


# ------------------- Bottleneck Residual Block -------------------
class BottleneckBlock(nn.Module):
    def __init__(self, channels, bottleneck_ratio=4):
        super().__init__()
        inner = channels // bottleneck_ratio
        self.conv1 = nn.Conv2d(channels, inner, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(inner)
        self.conv2 = nn.Conv2d(inner, inner, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(inner)
        self.conv3 = nn.Conv2d(inner, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += residual
        x = F.relu(x)
        return x


# ------------------- Channel Attention -------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, H, W]
        y = x.mean(dim=(2, 3))  # global average pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y[:, :, None, None]


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

    def policy_head_modules(self) -> list:
        return [self.policy_conv, self.policy_bn, self.policy_fc]

    def value_head_modules(self) -> list:
        return [self.value_conv, self.value_bn, self.value_fc1, self.value_fc2]

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


class CheckersNetV2(nn.Module):
    def __init__(
        self,
        input_planes,
        policy_planes,
        channels=192,
        num_blocks=24,
        temperature=1.2,
        use_attention=True,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_attention = use_attention

        # Initial convolution
        self.conv_input = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual trunk
        self.res_blocks = nn.ModuleList(
            [BottleneckBlock(channels) for _ in range(num_blocks)]
        )

        # Optional attention for long-range dependencies
        if self.use_attention:
            self.attention = ChannelAttention(channels)

        # ---------------- POLICY HEAD ----------------
        self.policy_conv = nn.Conv2d(
            channels, 8, kernel_size=1
        )  # more channels before flatten
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc1 = nn.Linear(8 * 8 * 8, 1024)
        self.policy_fc2 = nn.Linear(1024, policy_planes)

        # ---------------- VALUE HEAD ----------------
        self.value_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_fc3 = nn.Linear(128, 1)

    def policy_head_modules(self) -> list:
        return [self.policy_conv, self.policy_bn, self.policy_fc1, self.policy_fc2]

    def value_head_modules(self) -> list:
        return [
            self.value_conv,
            self.value_bn,
            self.value_fc1,
            self.value_fc2,
            self.value_fc3,
        ]

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual trunk
        for block in self.res_blocks:
            x = block(x)

        if self.use_attention:
            x = self.attention(x)

        # ================= POLICY =================
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = torch.flatten(p, start_dim=1)
        p = F.relu(self.policy_fc1(p))
        p = self.policy_fc2(p)
        p = p / self.temperature  # logits scaled by temperature

        # ================= VALUE =================
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = torch.flatten(v, start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = F.relu(self.value_fc2(v))
        v = torch.tanh(self.value_fc3(v))  # bounded evaluation

        return p, v


# ------------------- Example instantiation -------------------
# model = CheckersNetV2(
#     input_planes=18,  # your input feature planes
#     policy_planes=20480,  # size of your policy output
#     channels=192,
#     num_blocks=24,
#     temperature=1.2,
#     use_attention=True,
# )
