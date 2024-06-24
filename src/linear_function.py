def linear_function(theta0: float, theta1: float, mileage: int) -> float:
    """
    linear_function(theta0: float, theta1: float, mileage: int) -> float
    estimate the price of a car based on trained data (theta0, theta1) and
    user input (the mileage of the car) then return a predicted price
    """
    return theta0 + (theta1 * mileage)
