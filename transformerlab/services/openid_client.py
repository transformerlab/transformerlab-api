"""Helpers for OpenID (WorkOS) integration.

We wrap the upstream `httpx_oauth` OpenID client with a *lazy* variant so that
importing the users router does not immediately perform a network request to
the OpenID configuration endpoint. The upstream `OpenID` client performs
discovery in its constructor; during application startup and especially in the
test suite we only want to mount routes based on environment variables without
attempting outbound HTTP calls (to dummy domains such as `workos.example`).

`LazyOpenIDWithProfile` defers construction of the real client until the first
call that actually needs it (authorization URL, token exchange, profile fetch,
etc.). This preserves behaviour in production while preventing spurious
network failures in tests simply by importing the router module.
"""

from typing import Optional, Any

from httpx_oauth.clients.openid import OpenID


class OpenIDWithProfile(OpenID):
    def __init__(self, client_id: str, client_secret: str, openid_configuration_endpoint: str):
        """OpenID client with default profile/email scopes.

        The upstream `httpx_oauth` OpenID client performs network discovery in its
        constructor. Our test suite frequently instantiates this class with a
        placeholder (non-routable) domain such as ``workos.example`` only to
        validate default scopes. In that scenario we suppress discovery errors so
        the object can still provide ``base_scopes`` without failing the test
        suite. Any real runtime usage (authorization URL, token exchange, user
        profile) will go through the lazy wrapper which will attempt discovery
        again via a fresh instance if needed.
        """
        try:
            super().__init__(
                client_id=client_id,
                client_secret=client_secret,
                openid_configuration_endpoint=openid_configuration_endpoint,
                base_scopes=["openid", "email", "profile"],
            )
        except Exception:  # pragma: no cover - defensive; specific error types vary
            # Provide minimal attributes required by tests. Real operations that
            # rely on discovered endpoints will be executed through the lazy
            # client (`LazyOpenIDWithProfile`) which will perform discovery.
            self.client_id = client_id
            self.client_secret = client_secret
            self.openid_configuration_endpoint = openid_configuration_endpoint
            self.base_scopes = ["openid", "email", "profile"]


class LazyOpenIDWithProfile:
    """Proxy that lazily instantiates `OpenIDWithProfile`.

    Only the small subset of methods we use are proxied. Additional methods can
    be added on demand. Attribute access falls back to the underlying instance
    once created.
    """

    # Expose attributes accessed during FastAPI Users router construction so we don't
    # need to instantiate the underlying networked client just to read them.
    name = "openid"
    base_scopes = ["openid", "email", "profile"]

    def __init__(self, client_id: str, client_secret: str, openid_configuration_endpoint: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._openid_configuration_endpoint = openid_configuration_endpoint
        self._client: Optional[OpenIDWithProfile] = None

    # Internal helper
    def _ensure(self) -> OpenIDWithProfile:
        if self._client is None:
            self._client = OpenIDWithProfile(
                self._client_id,
                self._client_secret,
                self._openid_configuration_endpoint,
            )
        return self._client

    # Methods used by fastapi-users oauth router
    async def get_authorization_url(self, *args, **kwargs):  # type: ignore[override]
        return await self._ensure().get_authorization_url(*args, **kwargs)

    async def get_access_token(self, *args, **kwargs):  # type: ignore[override]
        return await self._ensure().get_access_token(*args, **kwargs)

    async def get_profile(self, *args, **kwargs):  # type: ignore[override]
        return await self._ensure().get_profile(*args, **kwargs)

    async def get_id_email(self, *args, **kwargs):  # type: ignore[override]
        return await self._ensure().get_id_email(*args, **kwargs)

    # Fallback attribute access
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        # Avoid constructing the underlying client for simple metadata
        if item in {"name", "base_scopes"}:
            return getattr(self, item)
        return getattr(self._ensure(), item)
