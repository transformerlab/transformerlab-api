import subprocess
import time
from typing import Optional


class GradioServerManager:
    """Manager for the Gradio server subprocess."""

    def __init__(self):
        """Initialize the manager with no running process."""
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None

    def start(self) -> subprocess.Popen:
        """Starts the gradio_server.py subprocess and returns its Popen object."""
        if self.is_running():
            print(f"Gradio server is already running with PID: {self.pid}")
            return self.process

        # Command to run the Gradio. We run it directly via python.
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "uvicorn",
            "transformerlab.services.gradio_server.gradio_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8339",
        ]
        print("Starting Gradio Server process...")

        # Using Popen to run the command in a new non-blocking process
        process = subprocess.Popen(command)
        self.process = process
        self.pid = process.pid

        print(f"Gradio Server process started with PID: {process.pid}")
        return process

    def stop(self) -> bool:
        """Stops the Gradio server subprocess."""
        if not self.is_running():
            print("No Gradio server is currently running.")
            return True

        try:
            print(f"Stopping Gradio Server with PID: {self.pid}")

            # Try graceful shutdown first
            self.process.terminate()

            # Wait for process to terminate gracefully
            try:
                self.process.wait(timeout=10)
                print("Gradio Server stopped gracefully.")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                print("Graceful shutdown timed out, forcing termination...")
                self.process.kill()
                self.process.wait()
                print("Gradio Server forcefully terminated.")

            self.process = None
            self.pid = None
            return True

        except Exception as e:
            print(f"Error stopping Gradio Server: {e}")
            return False

    def restart(self) -> subprocess.Popen:
        """Restarts the Gradio Server subprocess."""
        print("Restarting Gradio Server...")

        # Stop the current process
        self.stop()

        # Wait a moment before starting again
        time.sleep(1)

        # Start a new process
        return self.start()

    def is_running(self) -> bool:
        """Check if the Gradio server is currently running."""
        if self.process is None:
            return False

        # Check if process is still alive
        poll_result = self.process.poll()
        if poll_result is not None:
            # Process has terminated
            self.process = None
            self.pid = None
            return False

        return True

    def get_status(self) -> dict:
        """Get the current status of the Gradio server."""
        return {"is_running": self.is_running(), "pid": self.pid, "process": self.process is not None}


# Singleton instance - this will be created once and reused
_gradio_server_manager_instance: Optional[GradioServerManager] = None


def get_gradio_server_manager() -> GradioServerManager:
    """Get the singleton instance of GradioServerManager.

    Returns:
        GradioServerManager: The singleton instance
    """
    global _gradio_server_manager_instance
    if _gradio_server_manager_instance is None:
        _gradio_server_manager_instance = GradioServerManager()
    return _gradio_server_manager_instance


def reset_gradio_server_manager() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _gradio_server_manager_instance
    if _gradio_server_manager_instance is not None and _gradio_server_manager_instance.is_running():
        _gradio_server_manager_instance.stop()
    _gradio_server_manager_instance = None
