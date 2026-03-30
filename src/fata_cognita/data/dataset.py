"""PyTorch Dataset and DataLoader factory for trajectory data."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    """Dataset wrapping pre-built trajectory tensors.

    Args:
        static_features: Static encoder input, shape (N, F).
        life_states: Life state labels, shape (N, S).
        income: Log income values, shape (N, S).
        satisfaction: Satisfaction scores, shape (N, S).
        masks: Observation masks, shape (N, S).
        indices: Optional subset of row indices to include.
    """

    def __init__(
        self,
        static_features: torch.Tensor,
        life_states: torch.Tensor,
        income: torch.Tensor,
        satisfaction: torch.Tensor,
        masks: torch.Tensor,
        indices: list[int] | None = None,
    ) -> None:
        if indices is not None:
            idx = torch.tensor(indices, dtype=torch.long)
            self.static_features = static_features[idx]
            self.life_states = life_states[idx]
            self.income = income[idx]
            self.satisfaction = satisfaction[idx]
            self.masks = masks[idx]
        else:
            self.static_features = static_features
            self.life_states = life_states
            self.income = income
            self.satisfaction = satisfaction
            self.masks = masks

    def __len__(self) -> int:
        return len(self.static_features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "static_features": self.static_features[idx],
            "life_states": self.life_states[idx],
            "income": self.income[idx],
            "satisfaction": self.satisfaction[idx],
            "masks": self.masks[idx],
        }


def create_dataloaders(
    data: dict[str, torch.Tensor],
    splits: dict[str, list[int]],
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders from tensor data and split indices.

    Args:
        data: Dictionary with static_features, life_states, income,
            satisfaction, masks tensors.
        splits: Dictionary with 'train', 'val', 'test' index lists.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader instances.
    """
    loaders = {}
    for split_name, indices in splits.items():
        ds = TrajectoryDataset(
            static_features=data["static_features"],
            life_states=data["life_states"],
            income=data["income"],
            satisfaction=data["satisfaction"],
            masks=data["masks"],
            indices=indices,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders
