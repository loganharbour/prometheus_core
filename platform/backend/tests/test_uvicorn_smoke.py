# This file is part of Prometheus
# https://inl.gov/super-intelligence/prometheus/
#
# Copyright 2026 Battelle Energy Alliance, LLC
# All rights reserved, see NOTICE.txt for full restrictions
# https://github.com/idaholab/prometheus_core/blob/main/NOTICE.txt
#
# Licensed under the MIT License, please see LICENSE for details
# https://github.com/idaholab/prometheus_core/blob/main/LICENSE
#
# SPDX-License-Identifier: MIT

"""Boot a real ``uvicorn`` and talk to it over a socket.

Every other test drives the app through ``TestClient`` / direct ASGI calls, so none of
them touch uvicorn's own code: the HTTP parser, keep-alive handling, the ``--workers``
supervisor and signal handling. Without this, a uvicorn upgrade is only exercised by the
compose stack's ``/healthz`` check. These use the same CLI shape as
``docker-entrypoint.sh``; ``--lifespan off`` skips the admin-db / Valkey setup, since the
server is under test here, not the app.
"""

import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _uvicorn(workers: int) -> Iterator[str]:
    """Run uvicorn on a free port; yield its base URL; assert a graceful stop on SIGTERM."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--lifespan",
            "off",
        ],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            assert proc.poll() is None, f"uvicorn exited early:\n{proc.stdout.read()}"
            try:
                if httpx.get(f"{base}/healthz", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "uvicorn did not become ready in 30s"
            time.sleep(0.2)
        yield base
    finally:
        proc.terminate()  # SIGTERM: the supervisor must stop its workers and exit cleanly
        try:
            code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("uvicorn did not exit within 20s of SIGTERM")
        output = proc.stdout.read()
        # A lone server re-raises SIGTERM after cleaning up (status -15); the --workers
        # supervisor exits 0. Either is a clean stop; the log proves it was graceful.
        assert code in (0, -signal.SIGTERM), f"uvicorn exited {code} after SIGTERM:\n{output}"
        assert "Finished server process" in output, output


@pytest.mark.parametrize("workers", [1, 2])
def test_real_uvicorn_serves_requests_and_shuts_down_cleanly(workers):
    with _uvicorn(workers) as base:
        assert httpx.get(f"{base}/healthz").json() == {"status": "ok"}
        # Auth dependencies run behind the real HTTP parser too.
        assert httpx.get(f"{base}/api/events").status_code == 401


def test_real_uvicorn_keep_alive_and_connection_close():
    with _uvicorn(1) as base, httpx.Client(base_url=base) as client:
        # Two requests on one client reuse the connection.
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200
        # uvicorn 0.53 parses comma-separated, case-insensitive Connection tokens.
        resp = client.get("/healthz", headers={"Connection": "keep-alive, Close"})
        assert resp.status_code == 200
        assert resp.headers.get("connection", "").lower() == "close"
