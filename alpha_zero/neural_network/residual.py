from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from .abc import NeuralNetwork as AbstractNeuralNetwork


class ResidualBlock(nn.Module):
    conv_1: nn.Conv2d
    batch_norm_1: nn.BatchNorm2d
    conv_2: nn.Conv2d
    batch_norm_2: nn.BatchNorm2d

    def __init__(self, num_features: int = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            padding=1,
        )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=num_features)
        self.conv_2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            padding=1,
        )
        self.batch_norm_2 = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = F.relu(self.batch_norm_2(self.conv_1(x)) + residual)
        return x


class ResidualNetwork(nn.Module):
    action_size: int
    board_size: int

    start_block: nn.Sequential
    back_bone: nn.ModuleList
    policy_head: nn.Sequential
    value_head: nn.Sequential

    def __init__(
        self,
        action_size: int,
        board_size: int,
        num_features: int = 256,
        num_residual_blocks: int = 4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_size = action_size
        self.board_size = board_size

        self.start_block = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=num_features, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=num_features),
            nn.ReLU(),
        )
        self.back_bone = nn.ModuleList(
            [
                ResidualBlock(num_features=num_features)
                for _ in range(num_residual_blocks)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=32 * self.board_size * self.board_size,
                out_features=self.action_size,
            ),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=3, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=3 * self.board_size * self.board_size, out_features=1
            ),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.view(-1, 3, self.board_size, self.board_size)
        x = self.start_block(x)
        for residual_block in self.back_bone:
            x = residual_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class NeuralNetworkWrapper(AbstractNeuralNetwork):
    net: ResidualNetwork
    cuda: bool = torch.cuda.is_available()

    def __init__(self, action_size: int, board_size: int, *args, **kwargs) -> None:
        self.net = ResidualNetwork(
            action_size=action_size, board_size=board_size, *args, **kwargs
        )
        self.net = self.net.share_memory()

        if self.cuda:
            self.net = self.net.cuda()

    def load(self, filepath: Path) -> None:
        self.net = torch.load(filepath)

    def predict(self, canonical_state: np.ndarray) -> tuple[np.ndarray, float]:
        self.net.eval()
        state_tensor: Tensor = (
            torch.from_numpy(canonical_state).float().unsqueeze(dim=0)
        )
        if self.cuda:
            state_tensor = state_tensor.cuda()
        assert state_tensor.shape == (1, 3, self.net.board_size, self.net.board_size)
        policy: Tensor
        value: Tensor
        with torch.no_grad():
            policy, value = self.net(state_tensor)
        assert policy.shape == (1, self.net.action_size)
        assert value.shape == (1, 1)
        return torch.softmax(policy, dim=-1).squeeze(dim=0).cpu().numpy(), value.item()

    def save(self, filepath: Path) -> None:
        torch.save(self.net, filepath)

    def train(
        self,
        samples: list[tuple[np.ndarray, np.ndarray, float]],
        *,
        batch_size: int = 1024,
        epochs: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        self.net.train()
        optimizer: optim.Optimizer = optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

        for _ in range(epochs):
            batch_count: int = (len(samples) // batch_size) + 1
            for _ in range(batch_count):
                sample_indices: np.ndarray = np.random.choice(len(samples), batch_size)
                sample_boards: Iterable[np.ndarray]
                sample_policies: Iterable[np.ndarray]
                sample_values: Iterable[float]
                sample_boards, sample_policies, sample_values = zip(
                    *[samples[i] for i in sample_indices]
                )
                boards: Tensor = torch.from_numpy(np.array(sample_boards))
                target_policies: Tensor = torch.from_numpy(np.array(sample_policies))
                target_values: Tensor = torch.from_numpy(np.array(sample_values))
                if self.cuda:
                    boards = boards.cuda()
                    target_policies = target_policies.cuda()
                    target_values = target_values.cuda()
                optimizer.zero_grad()
                policies: Tensor
                values: Tensor
                policies, values = self.net(boards)
                loss: Tensor = F.mse_loss(values, target_values) + F.cross_entropy(
                    input=policies, target=target_policies
                )
                loss.backward()
                optimizer.step()
