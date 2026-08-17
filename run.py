"""Entrypoint that reads PORT from the environment directly in Python.

Railway injects PORT as an environment variable but does NOT guarantee shell
expansion in startCommand.  Reading it in Python makes the binding work
regardless of whether the process is launched via sh, exec, or any other
mechanism the platform uses internally.
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "copilot.api:app",
        host="0.0.0.0",
        port=port,
        timeout_graceful_shutdown=5,
    )
