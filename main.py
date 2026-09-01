"""
Entrypoint. Run with:

    python main.py

or, for auto-reload during development:

    uvicorn main:app --reload
"""

import uvicorn

from app import app  # noqa: F401  (re-exported so `uvicorn main:app` also works)

if __name__ == "__main__":
    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True)