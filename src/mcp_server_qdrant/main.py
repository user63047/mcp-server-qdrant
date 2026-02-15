import argparse


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml.

    Two modes:
    - stdio: MCP only (for local clients like Msty via STDIO)
    - streamable-http: Combined MCP + REST API on a single port
      (for container deployment with HTTP consumers)
    """

    parser = argparse.ArgumentParser(description="mcp-server-qdrant")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (HTTP mode only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (HTTP mode only)",
    )
    args = parser.parse_args()

    # Import here so environment variables are loaded after argument parsing
    from mcp_server_qdrant.server import mcp

    if args.transport == "stdio":
        # STDIO mode: MCP only, no REST API
        mcp.run(transport="stdio")
    else:
        # HTTP mode: combined MCP + REST API on single port
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        from mcp_server_qdrant.rest_api import create_rest_api

        # REST API shares the QdrantConnector with MCP server
        rest_app = create_rest_api(mcp.qdrant_connector)

        # MCP HTTP ASGI app (includes /mcp route internally)
        mcp_app = mcp.http_app()

        # Combined ASGI app: single port, path-based routing
        #   /mcp/    → MCP Streamable HTTP (from mcp_app's own routes)
        #   /api/v1/ → REST API (n8n sync flows)
        app = Starlette(
            routes=[
                *mcp_app.routes,
                Mount("/", app=rest_app),
            ],
            lifespan=mcp_app.lifespan,
        )

        # Fix: rewrite /mcp → /mcp/ for clients that strip trailing slash
        _inner = app

        async def app_wrapper(scope, receive, send):
            if scope["type"] == "http" and scope["path"] == "/mcp":
                scope = dict(scope, path="/mcp/")
            await _inner(scope, receive, send)

        uvicorn.run(app_wrapper, host=args.host, port=args.port)


