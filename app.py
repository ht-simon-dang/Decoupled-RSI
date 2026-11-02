#!/usr/bin/env python3
"""Local index and process launcher for every Decoupled RSI WebView.

Run ``python app.py``. The index uses only the Python standard library; each
Agent runs in its own installed environment and keeps its native API routes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    project: str
    module: str
    python_env: str
    port: int
    title: str
    description: str
    symbol: str
    category: str = "task"
    fragment: str = ""
    service: str = ""


AGENTS = (
    Agent("drug", "Drug Agent", "drug-agent", "drug_agent.llm.server", "DRUGAGENT_PYTHON", 2001,
          "LLM master agent — drug discovery", "从研究目标到小分子候选，追踪规划、生成与评价。", "D"),
    Agent("peptide", "Peptide Agent", "peptide-agent", "peptide_agent.web", "PEPTIDE_PYTHON", 2004,
          "Peptide Agent", "设计与筛选肽序列，查看结构、亲和力与计算证据。", "P"),
    Agent("mrna", "mRNA Agent", "mRNA-agent", "mrna_agent.web", "MRNA_PYTHON", 2005,
          "EMA4D mRNA Master Agent", "围绕目标蛋白设计构建体，比较翻译与稳定性证据。", "M"),
    Agent("enzyme", "Enzyme Agent", "Enzyme-Agent", "enzyme_agent.web", "ENZYME_PYTHON", 2006,
          "Enzyme Agent", "连接疾病机制、催化路线与酶工程候选。", "E"),
    Agent("antibody", "Antibody Agent", "antibody-agent", "antibody_agent.web", "ANTIBODY_PYTHON", 2007,
          "EMA4D Antibody Agent", "从靶点与表位出发，设计和评价抗体候选。", "Y"),
    Agent("self-evolve", "Self-Evolve Agent", "drug-agent", "drug_agent.llm.server", "DRUGAGENT_PYTHON", 2001,
          "LLM master agent — drug discovery", "查看执行经验、候选改进与独立评价记录。", "↻", "evolution", "evolve", "drug"),
    Agent("dashboard", "Drug Dashboard", "drug-agent", "drug_agent.api.server", "DRUGAGENT_PYTHON", 2000,
          "drug-agent API", "检查工作流、候选组合、风险与可复现报告。", "↗", "analysis"),
)
BY_ID = {agent.id: agent for agent in AGENTS}


def python_for(agent: Agent) -> str | None:
    """Prefer explicit settings, then the project's installed environments."""
    explicit = os.environ.get(agent.python_env)
    if explicit:
        candidate = Path(explicit).expanduser()
        return str(candidate.absolute()) if candidate.is_file() and os.access(candidate, os.X_OK) else shutil.which(explicit)
    candidates = [ROOT / "agents" / agent.project / ".venv/bin/python"]
    data_env = {"mrna": "MRNA_DATA_DIR", "enzyme": "ENZYME_DATA_DIR"}.get(agent.id)
    if data_env:
        data = Path(os.environ.get(data_env, Path.home() / ".cache/decoupled-rsi" / agent.id))
        candidates.extend([data / "env/bin/python", data / "venv/bin/python"])
        local = ROOT / ".local/dependencies" / agent.id
        candidates.extend([local / "env/bin/python", local / "venv/bin/python"])
    candidates.append(ROOT / ".local/environments" / agent.project / "bin/python")
    conda = shutil.which("conda")
    if conda:
        candidates.append(Path(conda).resolve().parent.parent / "envs" / agent.project / "bin/python")
    candidates.extend([Path(sys.prefix) / "bin/python", Path(sys.executable)])
    return next((str(p.absolute()) for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)


class Launcher:
    def __init__(self, *, port_offset: int = 0, data_dir: Path | None = None):
        self.port_offset = port_offset
        self.data_dir = data_dir
        self.processes: dict[str, subprocess.Popen] = {}
        self.locks = {a.id: threading.Lock() for a in AGENTS}
        self.closed = False

    def port(self, agent: Agent) -> int:
        return agent.port + self.port_offset

    def probe(self, agent: Agent) -> str:
        try:
            # Do not use configured HTTP proxies for loopback service checks.
            with build_opener(ProxyHandler({})).open(
                f"http://127.0.0.1:{self.port(agent)}/openapi.json", timeout=0.8
            ) as response:
                doc = json.load(response)
            return "running" if doc.get("info", {}).get("title") == agent.title else "conflict"
        except (URLError, OSError, ValueError):
            with socket.socket() as sock:
                sock.settimeout(0.2)
                return "conflict" if sock.connect_ex(("127.0.0.1", self.port(agent))) == 0 else "stopped"

    def status(self, agent: Agent) -> dict:
        service = agent.service or agent.id
        process = self.processes.get(service)
        state = self.probe(agent)
        if state == "stopped" and process and process.poll() is not None:
            state = "failed"
        return {"id": agent.id, "name": agent.name, "description": agent.description,
                "symbol": agent.symbol, "category": agent.category, "fragment": agent.fragment,
                "port": self.port(agent), "state": state, "managed": bool(process and process.poll() is None),
                "python_env": agent.python_env}

    def statuses(self) -> list[dict]:
        with ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
            return list(pool.map(self.status, AGENTS))

    def start(self, agent: Agent) -> dict:
        service = BY_ID[agent.service or agent.id]
        with self.locks[service.id]:
            if self.closed:
                raise RuntimeError("入口服务正在关闭。")
            state = self.probe(service)
            if state == "running":
                process = self.processes.get(service.id)
                if self.data_dir and (not process or process.poll() is not None):
                    raise RuntimeError("独立预览不能复用外部服务的数据目录。请使用 --port-offset 选择空闲端口。")
                return self.status(agent)
            if state == "conflict":
                raise RuntimeError(f"端口 {self.port(service)} 已被其他服务占用。请使用 --port-offset 调整 Agent 端口。")
            interpreter = python_for(service)
            if not interpreter:
                raise RuntimeError(f"未找到运行环境。请安装 {service.project} 或设置 {service.python_env}。")
            log_dir = (self.data_dir or ROOT / ".local/web-hub") / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{service.id}.log"
            command = [interpreter, str(ROOT / "app.py"), "--worker", service.id,
                       "--port", str(self.port(service))]
            if self.data_dir:
                command.extend(["--data-dir", str(self.data_dir)])
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "agents" / service.project / "src") + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with log_path.open("ab") as log:
                process = subprocess.Popen(command, cwd=ROOT / "agents" / service.project,
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            self.processes[service.id] = process
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not self.closed:
                if process.poll() is not None:
                    raise RuntimeError(f"{service.name} 启动失败。请检查 {service.python_env} 和日志 {log_path}。")
                if self.probe(service) == "running":
                    return self.status(agent)
                time.sleep(0.25)
            process.terminate()
            raise RuntimeError(f"{service.name} 启动超时，请检查日志 {log_path}。")

    def close(self):
        self.closed = True
        # Synchronize with in-flight starts; never terminate a reused service.
        for agent in AGENTS:
            with self.locks[agent.id]:
                process = self.processes.get(agent.id)
                if process and process.poll() is None:
                    process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class HubServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, launcher: Launcher):
        self.launcher = launcher
        super().__init__(address, HubHandler)


class HubHandler(BaseHTTPRequestHandler):
    server: HubServer

    def respond(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The browser may leave while a status probe is finishing.

    def json_response(self, status: int, data):
        self.respond(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def allowed(self, mutation=False) -> bool:
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("http://" + host)
            valid = parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == self.server.server_port
        except ValueError:
            valid = False
        if not valid:
            self.json_response(403, {"detail": "仅允许本机访问。"})
            return False
        origin = self.headers.get("Origin")
        if mutation and ((origin and origin != "http://" + host) or self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self.json_response(403, {"detail": "不接受跨站启动请求。"})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/agents":
            self.json_response(200, self.server.launcher.statuses())
            return
        # Explicit allowlist: never expose repository files or runtime data.
        assets = {"/": "index.html", "/index.php": "index.html", "/index.html": "index.html",
                  "/hub.css": "hub.css", "/hub.js": "hub.js",
                  "/shared.css": "shared.css", "/shared.js": "shared.js"}
        name = assets.get(path)
        if not name:
            self.json_response(404, {"detail": "页面不存在。"})
            return
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.respond(200, (WEB / name).read_bytes(), mime + "; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if not self.allowed(mutation=True):
            return
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "agents"] or parts[3] != "start" or parts[2] not in BY_ID:
            self.json_response(404, {"detail": "未知 Agent。"})
            return
        try:
            self.json_response(200, self.server.launcher.start(BY_ID[parts[2]]))
        except (OSError, RuntimeError) as exc:
            self.json_response(503, {"detail": str(exc)})


def run_worker(agent: Agent, port: int, data_dir: Path | None):
    """Imported only inside the selected Agent's Python environment."""
    import uvicorn

    if data_dir and agent.project == "drug-agent":
        for key, directory in {"DRUGAGENT_HOME": "drug", "DRUGAGENT_JOURNAL_DIR": "history",
                               "DRUGAGENT_EVOLVE_STORE_DIR": "evolve", "DRUGAGENT_EVOLVE_WORKSPACE_DIR": "workspaces"}.items():
            os.environ[key] = str(data_dir / directory)
        os.environ["DRUGAGENT_RECORD_HISTORY"] = "0"
    module = importlib.import_module(agent.module)
    if agent.id == "dashboard":
        application = module.app
    elif agent.id == "drug":
        application = module.create_app()
    else:
        application = module.create_app(data_dir / agent.id) if data_dir else module.create_app()
    uvicorn.run(application, host="127.0.0.1", port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="Index port (default: 8080)")
    parser.add_argument("--port-offset", type=int, default=0, help="Offset all Agent ports")
    parser.add_argument("--data-dir", type=Path, help="Use separate run stores for previews/testing")
    parser.add_argument("--open", action="store_true", help="Open the index in your browser")
    parser.add_argument("--worker", choices=[a.id for a in AGENTS if not a.service], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or any(not 1 <= a.port + args.port_offset <= 65535 for a in AGENTS):
        parser.error("Ports must be between 1 and 65535")
    if not args.worker and args.port in {a.port + args.port_offset for a in AGENTS}:
        parser.error("Index port overlaps an Agent port; use --port or --port-offset")
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else None
    if args.worker:
        run_worker(BY_ID[args.worker], args.port, data_dir)
        return
    launcher = Launcher(port_offset=args.port_offset, data_dir=data_dir)
    server = HubServer(("127.0.0.1", args.port), launcher)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Decoupled RSI · Agent Workspace\n{url}\nCtrl+C stops services started by this index.", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(url)
    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        launcher.close()
        server.server_close()


if __name__ == "__main__":
    main()
