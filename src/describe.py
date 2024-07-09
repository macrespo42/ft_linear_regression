from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy._typing import NDArray


@dataclass
class TrainingData:
    """TrainingData used to store and manipulate
    training model datas."""

    x: np.ndarray
    y: np.ndarray
    x_mean: np.floating[Any] = field(init=False)
    y_mean: np.floating[Any] = field(init=False)
    x_std: np.floating[Any] = field(init=False)
    y_std: np.floating[Any] = field(init=False)

    def __post_init__(self):
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)

        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)

    def normalize(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """normalize(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
        normalize data using z-score algorithm
        """
        x_normalized: NDArray[np.floating[Any]] = (self.x - self.x_mean) / self.x_std
        y_normalized: NDArray[np.floating[Any]] = (self.y - self.y_mean) / self.y_std

        return (x_normalized, y_normalized)

    def denormalize(self, theta0: float, theta1: float) -> tuple[float, float]:
        """def denormalize(self, theta0: float, theta1: float) -> tuple[np.floating, np.floating]
        denormalize thetas using z-score algorithm
        """
        d_theta1 = theta1 * (self.y_std / self.x_std)
        d_theta0 = (theta0 * self.y_std) + self.y_mean - d_theta1 * self.x_mean

        return float(d_theta0), float(d_theta1)
