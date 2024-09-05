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


@router.get("/prompt", summary="Returns a default system prompt containing a list of available tools")
async def get_prompt():
      tool_description_string = ""
      for name, func in available_tools.items():
          tool_description_string += f"{name}:\n{func.__doc__}\n\n"


      return f"""You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
Here are the available tools:
<tools>
{tool_description_string}
</tools>
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": <function-name>, "arguments": <args-dict>}}
</tool_call>
""";


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

        print("Successfully called", tool_id)
        print(result)
        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        err_string = f"{type(e).__name__}: {e}"
        print(err_string)
        return {
            "status": "error",
            "message": err_string
        }
