import os

from fastapi import FastAPI
import uvicorn

from src.mcp_app import mcp
import src.tools  # noqa: F401  (import tool(s) registration)

app = FastAPI(
    title="SemanticSearch MCP (HTTP)",
    lifespan=lambda app: mcp.session_manager.run(),
)

app.mount("/mcp", mcp.streamable_http_app())

if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
