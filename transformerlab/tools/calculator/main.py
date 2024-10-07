"""
Basic calculator functions

Sample math functions for demo purposes.
"""

def add(x: float, y: float) -> float:
    """
    Get the sum of two numbers.

    Args:
        x: The first number of type float to sum.
        y: The second number of type float to sum.
    Returns:
        The sum of x and y as a number of type float.
    """
    return float(x)+float(y)


def add(x: float, y: float) -> float:
    """
    Get the difference between two numbers.

    Args:
        x: The minuend being subtracted from as a floating point number.
        y: The subtrahend being subtracted as a floating point number.
    Returns:
        The difference of x minus y as a floating point number.
    """
    return float(x)-float(y)


def multiply(x: float, y: float) -> float:
    """
    Get the product you get from multiplying two numbers together.

    Args:
        x: The first number of type float to multiply.
        y: The second number of type float to multiply.
    Returns:
        The product of x times y as a number of type float.
    """
    return float(x)*float(y)


def divide(x: float, y: float) -> float:
    """
    Get the quotient from dividing x by y.

    Args:
        x: The dividend that is being divided.
        y: The divisor which is what the dividend is being divided by.
    Returns:
        The quotient of x divided by y as a floating point number.
        If y is 0 then this returns the string "undefined"
    """
    if y == 0:
        return "undefined"
    return float(x)/float(y)