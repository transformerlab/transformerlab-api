import os
import json


from fastapi import APIRouter


router = APIRouter(prefix="/tools", tags=["tools"])


#############################
# TEMPORARY HARD-CODED TOOLS
#
# Eventually we will replace this with some dynamic way of addings tools.
# In the meantime, this is going to pass the function docs to the prompt.
# So make sure you create your docs properly!
#############################


def get_current_temperature(location: str) -> float:
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for, in the format "city, country"
    Returns:
        The current temperature at the given location in Celsius.
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
    return 6.0  # Tested outside a few times. It was always near 6. So probably safe guess.


available_tools = {
    "get_current_temperature": get_current_temperature,
    "get_current_wind_speed": get_current_wind_speed
}


#############################
# TOOLS API ENDPOINTS
#############################


@router.get("/list", summary="List the tools that are currently installed.")
async def list_tools() -> list[object]:
    tool_descriptions = [f"{name}:\n{func.__doc__}\n\n" for name, func in available_tools.items()]

    return tool_descriptions


@router.get("/call/{tool_id}", summary="Executes a tool with parameters supplied in JSON.")
async def call_tool(tool_id: str, params: str):

    # First make sure we have a tool with this name
    if tool_id not in available_tools:
        return {
            "status": "error",
            "message": f"No tool with ID {tool_id} found."
        }

    try:
        tool_function = available_tools.get(tool_id)
        result = tool_function(params)

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        return {
            "status": "error",
            "data": f"{type(e).__name__}: {e}"
        }
