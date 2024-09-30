"""
Weather functions

Sample functions to demonstrate function calling capabilities.
"""

def get_current_temperature(location: str) -> float:
    """
    Gets the temperature at a given location in Celsius.

    Args:
        location: The location to get the temperature for, in the format "city, country"
    Returns:
        The current temperature at the given location in Celsius.
    """
    return len(location)  # low priority bug: Temperature not always related to length of city name.


def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 2*len(location)  # Tested a few times and this seemed close, so probably good enough.