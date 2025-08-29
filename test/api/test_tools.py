def test_tools_list(client):
    resp = client.get("/tools/list")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list) or isinstance(resp.json(), dict)


# def test_tools_call(client):
#     # Test should expect error since MCP server doesn't exist or MCP client not installed
#     resp = client.get("/tools/call/add?params={}&mcp_server_file=test_server")
#     assert resp.status_code == 200
#     assert resp.json()["status"] == "error"


def test_tools_all(client):
    resp = client.get("/tools/all")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# def test_tools_call_invalid_tool(client):
#     resp = client.get("/tools/call/invalid_tool?params={}&mcp_server_file=test_server")
#     assert resp.status_code == 200
#     assert resp.json()["status"] == "error"


# def test_tools_call_invalid_params(client):
#     resp = client.get("/tools/call/add?params=not_a_json&mcp_server_file=test_server")
#     assert resp.status_code == 200
#     assert resp.json()["status"] == "error"


def test_tools_install_mcp_server_invalid_file(client):
    resp = client.get("/tools/install_mcp_server?server_name=/not/a/real/path.py")
    assert resp.status_code == 403
    assert resp.json()["status"] == "error"


def test_mcp_servers_list(client):
    """Test listing MCP servers"""
    resp = client.get("/tools/mcp_servers")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_mcp_servers_add_invalid(client):
    """Test adding MCP server with invalid parameters"""
    resp = client.post("/tools/mcp_servers?server_id=test&server_name=invalid_server")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


def test_mcp_servers_delete_nonexistent(client):
    """Test deleting non-existent MCP server"""
    resp = client.delete("/tools/mcp_servers/nonexistent")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_mcp_servers_tools_nonexistent(client):
    """Test listing tools from non-existent MCP server"""
    resp = client.get("/tools/mcp_servers/nonexistent/tools")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
