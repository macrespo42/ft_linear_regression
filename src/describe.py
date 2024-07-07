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

    x_max: float = field(init=False)
    x_min: float = field(init=False)

    y_max: float = field(init=False)
    y_min: float = field(init=False)

    def __post_init__(self):
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)

        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)

        self.x_max = self.x.max()
        self.x_min = self.x.min()

        self.y_max = self.y.max()
        self.y_min = self.y.min()

    def normalize(self) -> tuple[np.ndarray]:
        """normalize(self) -> tuple[np.ndarray]
        normalize data using z-score algorithm
        """
        # x_normalized = (self.x - self.x_mean) / self.x_std
        # y_normalized = (self.y - self.y_mean) / self.y_std

        x_normalized = (self.x - self.x_max) / (self.x_max - self.x_min)
        y_normalized = (self.y - self.y_max) / (self.y_max - self.y_min)

        return x_normalized, y_normalized

    def denormalize(self, theta0: float, theta1: float) -> tuple[float]:
        x_min_max = self.x_max - self.x_min
        y_min_max = self.y_max - self.y_min

        d_theta0 = (
            theta0 * y_min_max
            + self.y_min
            + (theta1 * self.x_min * (y_min_max)) / (x_min_max)
        )
        d_theta1 = theta1 * (y_min_max) / (x_min_max)
        return d_theta0, d_theta1
