import os
import sys
import json
import importlib
from transformers.utils import get_json_schema
from transformerlab.shared import dirs

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi.responses import JSONResponse
import subprocess

# MCP client imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

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
                    for attr in dir(pkg):
                        func = getattr(pkg, attr)

                        # Add any functions that has pydocs
                        if callable(func) and func.__doc__:
                            func_name = func.__name__
                            available_tools[func_name] = func

    return available_tools


#############################
# TOOLS API ENDPOINTS
#############################


@router.get("/list", summary="List the tools that are currently installed.")
async def list_tools(
    mcp_server_file: str = Query(None, description="MCP server file to include MCP tools"),
    mcp_args: Optional[str] = Query(None, description="Comma-separated args for MCP server"),
    mcp_env: Optional[str] = Query(None, description="JSON string for MCP server env"),
) -> list[object]:
    available_tools = load_tools()
    tool_descriptions = [
        {"name": name, "description": f"{name}:\n{func.__doc__}\n\n"} for name, func in available_tools.items()
    ]
    if mcp_server_file:
        args = mcp_args.split(",") if mcp_args and len(mcp_args) > 1 else None
        # env = json.loads(mcp_env) if mcp_env else None
        base_env = os.environ.copy()
        override_env = json.loads(mcp_env) if mcp_env else {}
        env = {**base_env, **override_env}
        mcp_tools = await mcp_list_tools(mcp_server_file, args=args, env=env)
        mcp_tools = mcp_tools.tools
        # If MCP returns a list of dicts of Tool objects, convert them to dicts
        if isinstance(mcp_tools, list):
            mcp_tools_converted = []
            for tool in mcp_tools:
                if not isinstance(tool, dict):
                    mcp_tools_converted.append(tool.model_dump())
                else:
                    mcp_tools_converted.append(tool)
            tool_descriptions.extend(mcp_tools_converted)

        elif isinstance(mcp_tools, dict) and mcp_tools.get("status") == "error":
            tool_descriptions.append({"name": "MCP_ERROR", "description": mcp_tools.get("message")})
    return tool_descriptions


@router.get("/prompt", summary="Returns a default system prompt containing a list of available tools")
async def get_prompt(
    mcp_server_file: str = Query(None, description="MCP server file to include MCP tools"),
    mcp_args: Optional[str] = Query(None, description="Comma-separated args for MCP server"),
    mcp_env: Optional[str] = Query(None, description="JSON string for MCP server env"),
):
    available_tools = load_tools()
    tool_descriptions = [get_json_schema(func) for name, func in available_tools.items()]
    if mcp_server_file:
        args = mcp_args.split(",") if mcp_args and len(mcp_args) > 1 else None
        base_env = os.environ.copy()
        override_env = json.loads(mcp_env) if mcp_env else {}
        env = {**base_env, **override_env}
        mcp_tools = await mcp_list_tools(mcp_server_file, args=args, env=env)
        mcp_tools = mcp_tools.tools
        # If MCP returns a list of dicts of Tool objects, convert them to dicts
        if isinstance(mcp_tools, list):
            mcp_tools_converted = []
            for tool in mcp_tools:
                if not isinstance(tool, dict):
                    mcp_tools_converted.append(tool.model_dump())
                else:
                    mcp_tools_converted.append(tool)
            tool_descriptions.extend(mcp_tools_converted)
        elif isinstance(mcp_tools, dict) and mcp_tools.get("status") == "error":
            tool_descriptions.append({"name": "MCP_ERROR", "description": mcp_tools.get("message")})
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
async def call_tool(
    tool_id: str,
    params: str,
    mcp_server_file: str = Query(None, description="MCP server file to call MCP tool"),
    mcp_args: Optional[str] = Query(None, description="Comma-separated args for MCP server (if needed)"),
    mcp_env: Optional[str] = Query(None, description="JSON string for MCP server env (if needed)"),
):
    available_tools = load_tools()
    if tool_id not in available_tools:
        if mcp_server_file:
            args = mcp_args.split(",") if mcp_args and len(mcp_args) > 1 else None
            # env = json.loads(mcp_env) if mcp_env else None
            base_env = os.environ.copy()
            override_env = json.loads(mcp_env) if mcp_env else {}
            env = {**base_env, **override_env}
            try:
                function_args = json.loads(params)
            except Exception:
                return {"status": "error", "message": "Invalid parameters provided."}
            result = await mcp_call_tool(mcp_server_file, tool_id, arguments=function_args, args=args, env=env)
            final_result = ""
            for content in result.content:
                content = content.model_dump()
                if isinstance(content, dict) and content.get("type") == "text":
                    final_result += f"\n {content.get('text')}"
                elif isinstance(content, dict) and content.get("type") == "json":
                    final_result += f"\n {str(content.get('json'))}"

            return {"status": "success", "data": final_result}
        else:
            return {"status": "error", "message": f"No tool with ID {tool_id} found."}
    else:
        try:
            function_args = json.loads(params)
        except Exception as e:
            err_string = f"{type(e).__name__}: {e}"
            print(err_string)
            print("Passed JSON parameter string:")
            print(params)
            return {"status": "error", "message": "Invalid parameters provided."}
        try:
            tool_function = available_tools.get(tool_id)
            result = tool_function(**function_args)
            print("Successfully called", tool_id)
            print(result)
            return {"status": "success", "data": result}
        except Exception as e:
            err_string = f"{type(e).__name__}: {e}"
            print(err_string)
            return {"status": "error", "message": "An internal error has occurred."}


class MCPServerParams(BaseModel):
    server_file: str
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None


class MCPCallParams(MCPServerParams):
    arguments: Optional[Dict[str, Any]] = None


def _get_stdio_server_params(server_file: str, args=None, env=None):
    # If server_file ends with .py, treat as file; else as module
    if server_file.endswith(".py"):
        cmd_args = [server_file] + (args or [])
    else:
        cmd_args = ["-m", server_file] + (args or [])
    # Always use 'python' and pass os.environ.copy() as env
    return StdioServerParameters(
        command="python",
        args=cmd_args,
        env=os.environ.copy(),
    )


async def mcp_list_tools(server_file: str, args=None, env=None):
    if not (ClientSession and StdioServerParameters and stdio_client):
        return {"status": "error", "message": "MCP client not installed."}
    params = _get_stdio_server_params(server_file, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


async def mcp_call_tool(server_file: str, tool_id: str, arguments=None, args=None, env=None):
    if not (ClientSession and StdioServerParameters and stdio_client):
        return {"status": "error", "message": "MCP client not installed."}
    params = _get_stdio_server_params(server_file, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_id, arguments=arguments or {})


@router.get("/install_mcp_server", summary="Install or check MCP server module or script.")
async def install_mcp_server(server_name: str = Query(..., description="Module name or full path to .py file")):
    env = os.environ.copy()
    # If it's a .py file, treat as a full file path and check if it exists
    if server_name.endswith(".py"):
        if os.path.isfile(server_name):
            return {"status": "success", "message": f"File '{server_name}' exists."}
        else:
            return JSONResponse(
                status_code=404, content={"status": "error", "message": f"File '{server_name}' not found."}
            )
    # Otherwise, try to pip install the module using uv pip
    try:
        result = subprocess.run(
            ["uv", "pip", "install", server_name],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if result.returncode == 0:
            return {"status": "success", "message": f"Successfully installed '{server_name}'.", "output": result.stdout}
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Failed to install '{server_name}'.", "output": result.stderr},
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
