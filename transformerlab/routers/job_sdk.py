from xmlrpc.server import SimpleXMLRPCDispatcher
from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse


class XMLRPCRouter:
    """
    A router for FastAPI that handles XML-RPC requests.
    """

    def __init__(self, prefix="/rpc"):
        self.dispatcher = SimpleXMLRPCDispatcher(allow_none=True, encoding=None)
        self.router = APIRouter(prefix=prefix)

        # Register the POST endpoint to handle XML-RPC requests
        self.router.add_api_route("", self.handle_xmlrpc, methods=["POST"], response_class=PlainTextResponse)

    def register_function(self, function, name=None):
        """
        Register a function to be exposed via XML-RPC.

        Args:
            function: The function to register
            name: The name to register it under (defaults to the function name)
        """
        return self.dispatcher.register_function(function, name)

    def register_instance(self, instance, allow_dotted_names=False):
        """
        Register an instance to be exposed via XML-RPC.

        Args:
            instance: The instance to register
            allow_dotted_names: Whether to allow dotted names for methods
        """
        return self.dispatcher.register_instance(instance, allow_dotted_names)

    def register_introspection_functions(self):
        """
        Register the introspection functions.
        """
        return self.dispatcher.register_introspection_functions()

    def register_multicall_functions(self):
        """
        Register the multicall functions.
        """
        return self.dispatcher.register_multicall_functions()

    async def handle_xmlrpc(self, request: Request) -> Response:
        """
        Handle an XML-RPC request.

        Args:
            request: The FastAPI request object

        Returns:
            A PlainTextResponse with the XML-RPC response
        """
        # Read the request body
        request_body = await request.body()

        # Process the request and get the response
        response = self.dispatcher._marshaled_dispatch(request_body, getattr(request, "_dispatch_method", None))

        # Return the response with the correct content type
        return Response(content=response, media_type="text/xml")


# Add this function to create and return the configured XML-RPC router
def get_xmlrpc_router(prefix="/job_sdk"):
    """
    Create and return a configured XML-RPC router that can be included in a FastAPI app.

    Args:
        prefix: The URL prefix for the XML-RPC endpoint

    Returns:
        A configured FastAPI router instance
    """
    # Create a new XML-RPC router
    xmlrpc_router = XMLRPCRouter(prefix=prefix)

    # Example returning a string
    def hello(name="World"):
        return f"Hello, {name}!"

    # Example returning a dictionary
    def get_user(user_id):
        # This could fetch from a database in a real application
        users = {
            1: {"name": "Alice", "email": "alice@example.com", "active": True},
            2: {"name": "Bob", "email": "bob@example.com", "active": False},
        }
        return users.get(user_id, {"error": "User not found"})

    # Example returning a list of objects
    def list_users():
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
        ]

    # Example returning nested complex data
    def get_project_data(project_id):
        if project_id == 123:
            return {
                "id": project_id,
                "name": "AI Model Training",
                "status": "running",
                "progress": 0.75,
                "tasks": [
                    {"id": 1, "name": "Data preprocessing", "complete": True},
                    {"id": 2, "name": "Model training", "complete": False},
                ],
                "metrics": {"accuracy": 0.92, "loss": 0.08},
                "created_at": "2025-04-01T12:00:00Z",
            }
        return {"error": "Project not found"}

    # Register all our functions
    xmlrpc_router.register_function(hello)
    xmlrpc_router.register_function(get_user)
    xmlrpc_router.register_function(list_users)
    xmlrpc_router.register_function(get_project_data)

    # Enable introspection (optional)
    xmlrpc_router.register_introspection_functions()

    # Return the router property
    return xmlrpc_router.router
