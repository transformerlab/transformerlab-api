import importlib.util
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gradio import mount_gradio_app
import transformerlab.db as db  # noqa: F401 we need to load the db before dirs, or you get a circular import error
from transformerlab.shared import dirs


def discover_gradio_ui_files():
    """
    Discover gradio_ui.py files in plugin directories and return their paths.

    Returns:
        list: List of tuples containing (plugin_name, gradio_ui_file_path)
    """
    gradio_ui_files = []

    if not os.path.exists(dirs.PLUGIN_DIR):
        print(f"Plugin directory not found: {dirs.PLUGIN_DIR}")
        return gradio_ui_files

    print(f"Scanning for Gradio UIs in: {dirs.PLUGIN_DIR}")

    # Iterate through each directory in the plugin directory
    for plugin_name in os.listdir(dirs.PLUGIN_DIR):
        plugin_path = os.path.join(dirs.PLUGIN_DIR, plugin_name)

        # Skip if not a directory
        if not os.path.isdir(plugin_path):
            continue

        gradio_ui_path = os.path.join(plugin_path, "gradio_ui.py")

        # Check if gradio_ui.py exists
        if os.path.exists(gradio_ui_path):
            print(f"Found gradio_ui.py in plugin: {plugin_name}")
            gradio_ui_files.append((plugin_name, gradio_ui_path))

    return gradio_ui_files


def mount_gradio_app_from_file(app: FastAPI, gradio_ui_path: str, path: str = "/"):
    """
    Mount a Gradio app at the specified path.

    Args:
        app (FastAPI): The FastAPI application instance.
        gradio_ui_path (str): Path to the gradio_ui.py file.
        path (str): Path to mount the Gradio app.
    """
    if not os.path.exists(gradio_ui_path):
        raise FileNotFoundError(f"Gradio UI file not found: {gradio_ui_path}")

    try:
        # Import the Gradio app from the specified path
        spec = importlib.util.spec_from_file_location("gradio_app", gradio_ui_path)
        gradio_app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gradio_app_module)

        # Get the Gradio app and mount it properly
        gradio_app = gradio_app_module.app

        # Don't disable the queue - let Gradio handle it properly
        # Just ensure it's properly configured
        if not hasattr(gradio_app, "_queue") or gradio_app._queue is None:
            # Initialize the queue if it doesn't exist
            pass  # Let Gradio handle queue initialization

        # Mount the Gradio app
        mount_gradio_app(app, gradio_app, path=path)
        print(f"Successfully mounted Gradio app at {path}")

    except Exception as e:
        print(f"Error mounting Gradio app from {gradio_ui_path}: {e}")
        import traceback

        traceback.print_exc()
        raise

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("Starting Gradio Server...")

    # Discover and mount Gradio UIs on startup
    gradio_ui_files = discover_gradio_ui_files()

    if not gradio_ui_files:
        print("No Gradio UIs found in plugins")
    else:
        for plugin_name, gradio_ui_path in gradio_ui_files:
            try:
                # Mount the Gradio app at /{plugin_name}
                mount_path = f"/{plugin_name}"
                mount_gradio_app_from_file(app, gradio_ui_path, path=mount_path)
                print(f"Successfully mounted Gradio UI from plugin '{plugin_name}' at {mount_path}")
            except Exception as e:
                print(f"Failed to mount Gradio UI from plugin '{plugin_name}': {e}")
                import traceback

                traceback.print_exc()

    print("Gradio Server startup complete")
    yield

    print("Shutting down Gradio Server...")


# Create FastAPI app
app = FastAPI(
    title="TransformerLab Gradio Server",
    description="A server that automatically discovers and serves Gradio UIs from plugins",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint that lists available Gradio UIs"""
    gradio_ui_files = discover_gradio_ui_files()

    if not gradio_ui_files:
        return {
            "message": "Welcome to TransformerLab Gradio Server",
            "status": "No Gradio UIs found",
            "available_uis": [],
        }

    available_uis = [
        {"plugin_name": plugin_name, "url": f"/{plugin_name}", "description": f"Gradio UI for {plugin_name} plugin"}
        for plugin_name, _ in gradio_ui_files
    ]

    return {
        "message": "Welcome to TransformerLab Gradio Server",
        "status": f"Found {len(gradio_ui_files)} Gradio UI(s)",
        "available_uis": available_uis,
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


def run_server(host: str = "0.0.0.0", port: int = 8339):
    """Run the Gradio server"""
    print(f"Starting Gradio Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TransformerLab Gradio Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8339, help="Port to bind to")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
