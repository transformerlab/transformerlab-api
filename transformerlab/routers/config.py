import json
from fastapi import APIRouter
import transformerlab.db.db as db


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/get/{key}", summary="")
async def config_get(key: str):
    # Special handling for MCP_SERVER to return all configured servers
    if key == "MCP_SERVER":
        servers = {}

        # Get the legacy single server config
        legacy_config = await db.config_get(key=key)
        if legacy_config:
            try:
                servers["default"] = json.loads(legacy_config)
            except Exception:
                pass

        # Get all MCP_SERVER_* configurations
        mcp_configs = await db.config_get_all_with_prefix("MCP_SERVER_")
        for config_key, value in mcp_configs.items():
            server_id = config_key.replace("MCP_SERVER_", "")
            try:
                servers[server_id] = json.loads(value)
            except Exception:
                continue

        return servers
    else:
        # Normal config getting
        value = await db.config_get(key=key)
        return value


@router.get("/set", summary="")
async def config_set(k: str, v: str):
    # Special handling for MCP_SERVER to support multiple servers
    if k == "MCP_SERVER":
        # Parse the value to get server name
        try:
            config = json.loads(v)
            server_name = config.get("serverName", "")

            # Generate a unique server ID based on the server name
            if server_name == "markitdown_mcp":
                server_id = "markitdown"
            elif server_name == "mcp-server-filesystem":
                server_id = "filesystem"
            else:
                # Use the server name as the ID, but clean it up
                server_id = server_name.replace("-", "_").replace("_mcp", "").replace("mcp-server-", "")

            # Store with the new multi-server key format only (no legacy key)
            new_key = f"MCP_SERVER_{server_id}"
            await db.config_set(key=new_key, value=v)

            return {"key": new_key, "value": v, "server_id": server_id}
        except Exception as e:
            # Fall back to original behavior if parsing fails
            await db.config_set(key=k, value=v)
            return {"key": k, "value": v}
    else:
        # Normal config setting
        await db.config_set(key=k, value=v)
        return {"key": k, "value": v}
