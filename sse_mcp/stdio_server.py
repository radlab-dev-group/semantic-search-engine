from src.mcp_app import mcp

import src.tools  # noqa: F401  (import tool(s) registration)

if __name__ == "__main__":
    mcp.run(transport="stdio")
