import os

import uvicorn

uvicorn.run("mlx_server.app:app", host=os.getenv("MLX_HOST", "127.0.0.1"),
            port=int(os.getenv("MLX_PORT", "11435")), log_level="warning")
