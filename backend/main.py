from __future__ import annotations

import uvicorn

from .api import app
from .config import settings


def run() -> None:
    config = settings()
    uvicorn.run(app(config), host=config.host, port=config.port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    run()
