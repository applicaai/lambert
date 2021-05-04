from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from transformers.models.t5.modeling_t5 import T5Attention


NUM_BUCKETS = 32
relative_position_bucket = T5Attention._relative_position_bucket


class RelativeBiasBase(nn.Module, ABC):
    def __init__(self, num_heads: int, scaling_factor: float, max_distance: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.biases = nn.Embedding(NUM_BUCKETS, self.num_heads)

    @abstractmethod
    def get_positions(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_relative_positions(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        positions = self.get_positions(input_ids, bboxes)  # B x L
        query_positions = positions.unsqueeze(2)  # B x L x 1
        key_positions = positions.unsqueeze(1)  # B x 1 x L
        return (key_positions - query_positions) * self.scaling_factor  # B x L x L

    def get_buckets(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        relative_positions = self.compute_relative_positions(input_ids, bboxes).to(dtype=torch.long)  # B x L x L
        return relative_position_bucket(relative_positions, max_distance=self.max_distance, num_buckets=NUM_BUCKETS)

    def forward(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        buckets = self.get_buckets(input_ids, bboxes)  # B x L x L
        biases = self.biases(buckets)  # B x L x L x H
        return biases.permute([0, 3, 1, 2])  # B x H x L x L


class RelativeSequentialBias(RelativeBiasBase):
    def get_positions(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        return positions.unsqueeze(0)  # add dummy batch dimension; this will work through broadcasting


class RelativeHorizontalBias(RelativeBiasBase):
    def get_positions(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        return bboxes[:, :, 0]


class RelativeVerticalBias(RelativeBiasBase):
    def get_positions(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        return bboxes[:, :, [1, 3]].mean(dim=-1)


class RelativeBias(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.sequential_bias = RelativeSequentialBias(num_heads, scaling_factor=1.0, max_distance=128)
        self.horizontal_bias = RelativeHorizontalBias(num_heads, scaling_factor=100, max_distance=64)
        self.vertical_bias = RelativeVerticalBias(num_heads, scaling_factor=100, max_distance=64)

    def forward(self, input_ids: Tensor, bboxes: Tensor) -> Tensor:
        sequential_bias = self.sequential_bias(input_ids, bboxes)
        horizontal_bias = self.horizontal_bias(input_ids, bboxes)
        vertical_bias = self.vertical_bias(input_ids, bboxes)
        return sequential_bias + horizontal_bias + vertical_bias
