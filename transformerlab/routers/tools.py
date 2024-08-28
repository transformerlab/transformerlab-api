import os
import json


from fastapi import APIRouter


router = APIRouter(prefix="/tools", tags=["tools"])


#############################
# TEMPORARY HARD-CODED TOOLS
#############################


def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for, in the format "city, country"
    """
    return 22.0  # bug: Sometimes the temperature is not 22. low priority to fix tho

def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!


#############################
# TOOLS API ENDPOINTS
#############################


@router.get("/list", summary="List the tools that are currently installed.")
async def list_tools() -> list[object]:
    result = []
    return result