from typing import Optional
from urllib.parse import quote

from fastapi import Response
from fastapi_users.authentication import BearerTransport
from starlette.responses import RedirectResponse


class RedirectBearerTransport(BearerTransport):
    """
    Bearer transport that, on successful login, redirects the browser to
    the configured frontend URL and passes the token via URL fragment.

    This avoids rendering a JSON token response in the browser during
    OAuth callbacks and hands control back to the frontend app.
    """

    def __init__(self, tokenUrl: str, frontend_redirect_url: str, *, token_param_name: str = "access_token"):
        super().__init__(tokenUrl=tokenUrl)
        self.frontend_redirect_url = frontend_redirect_url.rstrip("/")
        self.token_param_name = token_param_name

    async def get_login_response(self, token: str, response: Optional[Response] = None):  # type: ignore[override]
        # Send token as URL fragment to avoid leaking it via server logs
        # or intermediaries on the frontend origin.
        encoded_token = quote(token, safe="")
        redirect_to = (
            f"{self.frontend_redirect_url}#{self.token_param_name}={encoded_token}&token_type=bearer"
        )
        return RedirectResponse(url=redirect_to, status_code=302)
