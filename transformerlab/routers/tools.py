import os
import sys
import json
import importlib
from transformers.utils import get_json_schema
from transformerlab.shared import dirs

from fastapi import APIRouter


router = APIRouter(prefix="/tools", tags=["tools"])


###################################################################
# Tools loading function
#
# This should be called by every tools endpoint that needs
# access to the list of available tools.
#
# NOTE: Adding tools right out of API directory.
#   This should be updated to copy and then look in workspace.
####################################################################


def load_tools():
    available_tools = {}

    # TODO: Pull dir from dirs.py
    tools_dir = os.path.join(dirs.TFL_SOURCE_CODE_DIR, "transformerlab", "tools")
    sys.path.append(tools_dir)

    # Scan the tools dir to subdirectories
    with os.scandir(tools_dir) as dirlist:
        for entry in dirlist:
            if entry.is_dir():

                # Try importing main.py from the subdirectory
                package_name = entry.name
                import_name = f"{package_name}.main"
                pkg = importlib.import_module(import_name)

                # Go through package contents and look for functions
                if pkg:
                    print(f"Importing tool package {package_name}")
                    for attr in dir(pkg):
                        func = getattr(pkg, attr)

                        # Add any functions that has pydocs
                        if callable(func) and func.__doc__:
                            func_name = func.__name__
                            print(f"Adding tool: {func_name}")
                            available_tools[func_name] = func

    return available_tools


#############################
# TOOLS API ENDPOINTS
#############################


@router.get("/list", summary="List the tools that are currently installed.")
async def list_tools() -> list[object]:
    available_tools = load_tools()

    tool_descriptions = []
    for name, func in available_tools.items():
        tool = {
            "name": name,
            "description": f"{name}:\n{func.__doc__}\n\n"
        }
        tool_descriptions.append(tool)

    return tool_descriptions


@router.get("/prompt", summary="Returns a default system prompt containing a list of available tools")
async def get_prompt():
    available_tools = load_tools()

    # Follow the format described here: https://huggingface.co/blog/unified-tool-use
    # Otherwise the models don't respond correclty
    tool_descriptions = []
    for name, func in available_tools.items():
        tool_descriptions.append(get_json_schema(func))
    tool_json = json.dumps(tool_descriptions)

    return f"""You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
Here are the available tools:
<tools>
{tool_json}
</tools>
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": <function-name>, "arguments": <args-dict>}}
</tool_call>
"""


@router.get("/call/{tool_id}", summary="Executes a tool with parameters supplied in JSON.")
async def call_tool(tool_id: str, params: str):
    available_tools = load_tools()

    # First make sure we have a tool with this name
    if tool_id not in available_tools:
        return {
            "status": "error",
            "message": f"No tool with ID {tool_id} found."
        }

    # Try to parse out parameters
    try:
        function_args = json.loads(params)
    except Exception as e:
        err_string = f"{type(e).__name__}: {e}"
        print(err_string)
        print("Passed JSON parameter string:")
        print(params)
        return {
            "status": "error",
            "message": err_string
        }

    try:
        tool_function = available_tools.get(tool_id)
        result = tool_function(**function_args)

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
