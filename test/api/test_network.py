"""
Tests for the network router endpoints.
"""

from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from api import app


class TestNetworkInfo:
    """Test the /network/info endpoint."""

    def test_get_network_info_host_machine(self):
        """Test getting network info when IS_HOST_MACHINE is True."""
        with patch("transformerlab.db.config_get") as mock_config_get:
            mock_config_get.return_value = "True"

            with TestClient(app) as client:
                response = client.get("/network/info")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"]["is_host_machine"] is True
                assert data["data"]["role"] == "host"

    def test_get_network_info_network_machine(self):
        """Test getting network info when IS_HOST_MACHINE is False."""
        with patch("transformerlab.db.config_get") as mock_config_get:
            mock_config_get.return_value = "False"

            with TestClient(app) as client:
                response = client.get("/network/info")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"]["is_host_machine"] is False
                assert data["data"]["role"] == "network_machine"

    def test_get_network_info_no_config(self):
        """Test getting network info when IS_HOST_MACHINE is not set."""
        with patch("transformerlab.db.config_get") as mock_config_get:
            mock_config_get.return_value = None

            with TestClient(app) as client:
                response = client.get("/network/info")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"]["is_host_machine"] is False
                assert data["data"]["role"] == "network_machine"

    def test_get_network_info_error(self):
        """Test error handling in network info endpoint."""
        with patch("transformerlab.db.config_get") as mock_config_get:
            mock_config_get.side_effect = Exception("Database error")

            with TestClient(app) as client:
                response = client.get("/network/info")

                assert response.status_code == 500
                assert "Internal server error" in response.json()["detail"]


class TestNetworkMachines:
    """Test the network machines CRUD endpoints."""

    def test_list_machines_empty(self):
        """Test listing machines when none exist."""
        with patch("transformerlab.db.network_machine_get_all") as mock_get_all:
            mock_get_all.return_value = []

            with TestClient(app) as client:
                response = client.get("/network/machines")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"] == []

    def test_list_machines_with_data(self):
        """Test listing machines with existing data."""
        mock_machines = [
            {
                "id": 1,
                "name": "machine1",
                "host": "192.168.1.100",
                "port": 8338,
                "status": "online",
                "last_seen": "2025-06-18T10:00:00",
                "metadata": {},
                "created_at": "2025-06-18T09:00:00",
                "updated_at": "2025-06-18T10:00:00",
            },
            {
                "id": 2,
                "name": "machine2",
                "host": "192.168.1.101",
                "port": 8338,
                "status": "offline",
                "last_seen": None,
                "metadata": {"gpu_count": 2},
                "created_at": "2025-06-18T09:30:00",
                "updated_at": "2025-06-18T09:30:00",
            },
        ]

        with patch("transformerlab.db.network_machine_get_all") as mock_get_all:
            mock_get_all.return_value = mock_machines

            with TestClient(app) as client:
                response = client.get("/network/machines")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert len(data["data"]) == 2
                assert data["data"] == mock_machines

    def test_list_machines_error(self):
        """Test error handling in list machines endpoint."""
        with patch("transformerlab.db.network_machine_get_all") as mock_get_all:
            mock_get_all.side_effect = Exception("Database error")

            with TestClient(app) as client:
                response = client.get("/network/machines")

                assert response.status_code == 500
                assert "Internal server error" in response.json()["detail"]

    def test_add_machine_success(self):
        """Test successfully adding a new machine."""
        machine_data = {
            "name": "test-machine",
            "host": "192.168.1.100",
            "port": 8338,
            "api_token": "test-token",
            "metadata": {"gpu_count": 4},
        }

        with (
            patch("transformerlab.db.network_machine_get_by_name") as mock_get_by_name,
            patch("transformerlab.db.network_machine_create") as mock_create,
            patch("transformerlab.routers.network.ping_machine") as mock_ping,
        ):
            mock_get_by_name.return_value = None  # No existing machine
            mock_create.return_value = 1  # New machine ID
            mock_ping.return_value = {"status": "online"}

            with TestClient(app) as client:
                response = client.post("/network/machines", json=machine_data)

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["machine_id"] == 1
                assert "test-machine" in data["message"]

                # Verify the database calls
                mock_get_by_name.assert_called_once_with("test-machine")
                mock_create.assert_called_once_with(
                    name="test-machine",
                    host="192.168.1.100",
                    port=8338,
                    api_token="test-token",
                    metadata={"gpu_count": 4},
                )
                mock_ping.assert_called_once_with(1)

    def test_add_machine_duplicate_name(self):
        """Test adding a machine with duplicate name."""
        machine_data = {"name": "existing-machine", "host": "192.168.1.100", "port": 8338}

        with patch("transformerlab.db.network_machine_get_by_name") as mock_get_by_name:
            mock_get_by_name.return_value = {"id": 1, "name": "existing-machine"}

            with TestClient(app) as client:
                response = client.post("/network/machines", json=machine_data)

                assert response.status_code == 400
                assert "already exists" in response.json()["detail"]

    def test_add_machine_minimal_data(self):
        """Test adding a machine with minimal required data."""
        machine_data = {"name": "minimal-machine", "host": "192.168.1.100"}

        with (
            patch("transformerlab.db.network_machine_get_by_name") as mock_get_by_name,
            patch("transformerlab.db.network_machine_create") as mock_create,
            patch("transformerlab.routers.network.ping_machine") as mock_ping,
        ):
            mock_get_by_name.return_value = None
            mock_create.return_value = 2
            mock_ping.return_value = {"status": "online"}

            with TestClient(app) as client:
                response = client.post("/network/machines", json=machine_data)

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["machine_id"] == 2

                # Verify defaults are applied
                mock_create.assert_called_once_with(
                    name="minimal-machine",
                    host="192.168.1.100",
                    port=8338,  # Default port
                    api_token=None,
                    metadata={},  # Default empty metadata
                )

    def test_add_machine_invalid_data(self):
        """Test adding a machine with invalid data."""
        with TestClient(app) as client:
            # Missing required field 'name'
            response = client.post("/network/machines", json={"host": "192.168.1.100"})
            assert response.status_code == 400

            # Missing required field 'host'
            response = client.post("/network/machines", json={"name": "test"})
            assert response.status_code == 400

            # Invalid port type
            response = client.post(
                "/network/machines", json={"name": "test", "host": "192.168.1.100", "port": "invalid"}
            )
            assert response.status_code == 400

    def test_get_machine_success(self):
        """Test successfully getting a specific machine."""
        mock_machine = {
            "id": 1,
            "name": "test-machine",
            "host": "192.168.1.100",
            "port": 8338,
            "status": "online",
            "last_seen": "2025-06-18T10:00:00",
            "metadata": {},
            "created_at": "2025-06-18T09:00:00",
            "updated_at": "2025-06-18T10:00:00",
        }

        with patch("transformerlab.db.network_machine_get") as mock_get:
            mock_get.return_value = mock_machine

            with TestClient(app) as client:
                response = client.get("/network/machines/1")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"] == mock_machine

    def test_get_machine_not_found(self):
        """Test getting a non-existent machine."""
        with patch("transformerlab.db.network_machine_get") as mock_get:
            mock_get.return_value = None

            with TestClient(app) as client:
                response = client.get("/network/machines/999")

                assert response.status_code == 404
                assert "Machine not found" in response.json()["detail"]

    def test_remove_machine_success(self):
        """Test successfully removing a machine."""
        with patch("transformerlab.db.network_machine_delete") as mock_delete:
            mock_delete.return_value = True

            with TestClient(app) as client:
                response = client.delete("/network/machines/1")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "removed successfully" in data["message"]

    def test_remove_machine_not_found(self):
        """Test removing a non-existent machine."""
        with patch("transformerlab.db.network_machine_delete") as mock_delete:
            mock_delete.return_value = False

            with TestClient(app) as client:
                response = client.delete("/network/machines/999")

                assert response.status_code == 404
                assert "Machine not found" in response.json()["detail"]

    def test_remove_machine_by_name_success(self):
        """Test successfully removing a machine by name."""
        with patch("transformerlab.db.network_machine_delete_by_name") as mock_delete:
            mock_delete.return_value = True

            with TestClient(app) as client:
                response = client.delete("/network/machines/by-name/test-machine")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "test-machine" in data["message"]

    def test_remove_machine_by_name_not_found(self):
        """Test removing a non-existent machine by name."""
        with patch("transformerlab.db.network_machine_delete_by_name") as mock_delete:
            mock_delete.return_value = False

            with TestClient(app) as client:
                response = client.delete("/network/machines/by-name/nonexistent")

                assert response.status_code == 404
                assert "Machine not found" in response.json()["detail"]


class TestNetworkStatus:
    """Test the network status endpoints."""

    def test_get_network_status_empty(self):
        """Test getting network status with no machines."""
        with patch("transformerlab.db.network_machine_get_all") as mock_get_all:
            mock_get_all.return_value = []

            with TestClient(app) as client:
                response = client.get("/network/status")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                status_data = data["data"]
                assert status_data["total_machines"] == 0
                assert status_data["online"] == 0
                assert status_data["offline"] == 0
                assert status_data["error"] == 0
                assert status_data["machines"] == []

    def test_get_network_status_with_machines(self):
        """Test getting network status with various machine states."""
        mock_machines = [
            {"id": 1, "name": "machine1", "status": "online"},
            {"id": 2, "name": "machine2", "status": "online"},
            {"id": 3, "name": "machine3", "status": "offline"},
            {"id": 4, "name": "machine4", "status": "error"},
            {"id": 5, "name": "machine5", "status": "offline"},
        ]

        with patch("transformerlab.db.network_machine_get_all") as mock_get_all:
            mock_get_all.return_value = mock_machines

            with TestClient(app) as client:
                response = client.get("/network/status")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                status_data = data["data"]
                assert status_data["total_machines"] == 5
                assert status_data["online"] == 2
                assert status_data["offline"] == 2
                assert status_data["error"] == 1
                assert status_data["machines"] == mock_machines


class TestPingMachine:
    """Test the machine ping functionality."""

    def test_ping_machine_endpoint_success(self):
        """Test the ping machine endpoint."""
        mock_result = {"status": "online", "response_time": 0.123, "server_info": {"version": "1.0.0"}}

        with patch("transformerlab.routers.network.ping_machine") as mock_ping:
            mock_ping.return_value = mock_result

            with TestClient(app) as client:
                response = client.post("/network/machines/1/ping")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"] == mock_result

    def test_ping_machine_endpoint_not_found(self):
        """Test pinging a non-existent machine."""
        with patch("transformerlab.routers.network.ping_machine") as mock_ping:
            from fastapi import HTTPException

            mock_ping.side_effect = HTTPException(status_code=404, detail="Machine not found")

            with TestClient(app) as client:
                response = client.post("/network/machines/999/ping")

                assert response.status_code == 404
                assert "Machine not found" in response.json()["detail"]

    @patch("transformerlab.routers.network.httpx.AsyncClient")
    @patch("transformerlab.db.network_machine_get")
    @patch("transformerlab.db.network_machine_update_status")
    @patch("transformerlab.db.network_machine_update_metadata")
    def test_ping_machine_function_online(self, mock_update_metadata, mock_update_status, mock_get, mock_client_class):
        """Test the ping_machine helper function when machine is online."""
        # Mock machine data
        mock_machine = {"id": 1, "host": "192.168.1.100", "port": 8338, "api_token": "test-token"}
        mock_get.return_value = mock_machine

        # Mock HTTP response - use MagicMock for sync methods
        mock_response = MagicMock()
        mock_response.json.return_value = {"version": "1.0.0", "status": "healthy"}
        mock_response.raise_for_status.return_value = None

        # Mock the async client properly
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        # Mock the async context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client
        mock_client_instance.__aexit__.return_value = None
        mock_client_class.return_value = mock_client_instance

        # Import and test the function directly
        from transformerlab.routers.network import ping_machine

        async def test_ping():
            result = await ping_machine(1)
            assert result["status"] == "online"
            assert "response_time" in result
            assert result["server_info"] == {"version": "1.0.0", "status": "healthy"}

            # Verify database updates
            mock_update_status.assert_called_once_with(1, "online")
            mock_update_metadata.assert_called_once()

        # Run the async test
        import asyncio

        asyncio.run(test_ping())

    @patch("transformerlab.routers.network.httpx.AsyncClient")
    @patch("transformerlab.db.network_machine_get")
    @patch("transformerlab.db.network_machine_update_status")
    def test_ping_machine_function_timeout(self, mock_update_status, mock_get, mock_client_class):
        """Test the ping_machine helper function when request times out."""
        mock_machine = {"id": 1, "host": "192.168.1.100", "port": 8338, "api_token": None}
        mock_get.return_value = mock_machine

        # Mock timeout exception
        import httpx

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("Request timeout")

        # Mock the async context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client
        mock_client_instance.__aexit__.return_value = None
        mock_client_class.return_value = mock_client_instance

        from transformerlab.routers.network import ping_machine

        async def test_ping():
            result = await ping_machine(1)
            assert result["status"] == "offline"
            assert result["error"] == "Request timeout"

            mock_update_status.assert_called_once_with(1, "offline")

        import asyncio

        asyncio.run(test_ping())

    @patch("transformerlab.routers.network.httpx.AsyncClient")
    @patch("transformerlab.db.network_machine_get")
    @patch("transformerlab.db.network_machine_update_status")
    def test_ping_machine_function_connection_error(self, mock_update_status, mock_get, mock_client_class):
        """Test the ping_machine helper function when connection fails."""
        mock_machine = {"id": 1, "host": "192.168.1.100", "port": 8338, "api_token": None}
        mock_get.return_value = mock_machine

        # Mock connection error
        import httpx

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        # Mock the async context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client
        mock_client_instance.__aexit__.return_value = None
        mock_client_class.return_value = mock_client_instance

        from transformerlab.routers.network import ping_machine

        async def test_ping():
            result = await ping_machine(1)
            assert result["status"] == "offline"
            assert result["error"] == "Server unreachable"

            mock_update_status.assert_called_once_with(1, "offline")

        import asyncio

        asyncio.run(test_ping())


class TestHealthCheck:
    """Test the health check functionality."""

    def test_health_check_all_success(self):
        """Test health check for all machines."""
        mock_machines = [{"id": 1, "name": "machine1"}, {"id": 2, "name": "machine2"}]

        mock_ping_results = [
            {"status": "online", "response_time": 0.1},
            {"status": "offline", "error": "Connection refused"},
        ]

        with (
            patch("transformerlab.db.network_machine_get_all") as mock_get_all,
            patch("transformerlab.routers.network.ping_machine") as mock_ping,
        ):
            mock_get_all.return_value = mock_machines
            mock_ping.side_effect = mock_ping_results

            with TestClient(app) as client:
                response = client.post("/network/health-check")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"

                results = data["data"]
                assert len(results) == 2

                # Check first machine result
                assert results[0]["machine_id"] == 1
                assert results[0]["name"] == "machine1"
                assert results[0]["status"] == "online"
                assert results[0]["response_time"] == 0.1

                # Check second machine result
                assert results[1]["machine_id"] == 2
                assert results[1]["name"] == "machine2"
                assert results[1]["status"] == "offline"

    def test_health_check_all_with_exceptions(self):
        """Test health check when some machines throw exceptions."""
        mock_machines = [{"id": 1, "name": "machine1"}, {"id": 2, "name": "machine2"}]

        with (
            patch("transformerlab.db.network_machine_get_all") as mock_get_all,
            patch("transformerlab.routers.network.ping_machine") as mock_ping,
        ):
            mock_get_all.return_value = mock_machines
            mock_ping.side_effect = [{"status": "online", "response_time": 0.1}, Exception("Network error")]

            with TestClient(app) as client:
                response = client.post("/network/health-check")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"

                results = data["data"]
                assert len(results) == 2

                # Check successful machine
                assert results[0]["status"] == "online"

                # Check failed machine
                assert results[1]["machine_id"] == 2
                assert results[1]["name"] == "machine2"
                assert results[1]["status"] == "error"
                assert "Network error" in results[1]["error"]

    def test_health_check_all_empty(self):
        """Test health check with no machines."""
        with patch("transformerlab.db.network_machine_get_all") as mock_get_all:
            mock_get_all.return_value = []

            with TestClient(app) as client:
                response = client.post("/network/health-check")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["data"] == []


# Integration-style tests that test multiple components together
class TestNetworkIntegration:
    """Integration tests for network functionality."""

    def test_full_machine_lifecycle(self):
        """Test the complete lifecycle of a machine: add, get, ping, remove."""
        machine_data = {"name": "lifecycle-test", "host": "192.168.1.100", "port": 8338}

        with (
            patch("transformerlab.db.network_machine_get_by_name") as mock_get_by_name,
            patch("transformerlab.db.network_machine_create") as mock_create,
            patch("transformerlab.db.network_machine_get") as mock_get,
            patch("transformerlab.db.network_machine_delete") as mock_delete,
            patch("transformerlab.routers.network.ping_machine") as mock_ping,
        ):
            # Setup mocks for add machine
            mock_get_by_name.return_value = None
            mock_create.return_value = 1
            mock_ping.return_value = {"status": "online"}

            # Setup mock for get machine
            mock_machine = {
                "id": 1,
                "name": "lifecycle-test",
                "host": "192.168.1.100",
                "port": 8338,
                "status": "online",
            }
            mock_get.return_value = mock_machine

            # Setup mock for delete
            mock_delete.return_value = True

            with TestClient(app) as client:
                # 1. Add machine
                response = client.post("/network/machines", json=machine_data)
                assert response.status_code == 200
                machine_id = response.json()["machine_id"]

                # 2. Get machine
                response = client.get(f"/network/machines/{machine_id}")
                assert response.status_code == 200
                assert response.json()["data"]["name"] == "lifecycle-test"

                # 3. Ping machine
                response = client.post(f"/network/machines/{machine_id}/ping")
                assert response.status_code == 200
                assert response.json()["data"]["status"] == "online"

                # 4. Remove machine
                response = client.delete(f"/network/machines/{machine_id}")
                assert response.status_code == 200
                assert "removed successfully" in response.json()["message"]
