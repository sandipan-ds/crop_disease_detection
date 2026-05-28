"""
Run the API locally for development/testing.

Usage:
    python -m api.run_local

Then visit: http://localhost:8000/docs for Swagger UI
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
