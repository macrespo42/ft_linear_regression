from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrainingData:
    """TrainingData used to store and manipulate
    training model datas."""

    x: np.ndarray
    y: np.ndarray
    x_mean: float = field(init=False)
    y_mean: float = field(init=False)
    x_std: float = field(init=False)
    y_std: float = field(init=False)

    def __post_init__(self):
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)

        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)

    def normalize(self) -> tuple[np.ndarray]:
        """normalize(self) -> tuple[np.ndarray]
        normalize data using z-score algorithm
        """
        x_normalized = (self.x - self.x_mean) / self.x_std
        y_normalized = (self.y - self.y_mean) / self.y_std

        return x_normalized, y_normalized

    def denormalize(self, theta0: float, theta1: float) -> tuple[float]:
        """def denormalize(self, theta0: float, theta1: float) -> tuple[float]
        denormalize thetas using z-score algorithm
        """
        d_theta1 = theta1 * (self.y_std / self.x_std)
        d_theta0 = (theta0 * self.y_std) + self.y_mean - d_theta1 * self.x_mean

        return d_theta0, d_theta1
