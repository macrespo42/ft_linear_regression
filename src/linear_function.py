def linear_function(theta0: float, theta1: float, x: float) -> float:
    """
    linear_function(theta0: float, theta1: float, x: int) -> float
    perform a linear regression (t1 * x) + t0
    """
    return theta0 + (theta1 * x)
