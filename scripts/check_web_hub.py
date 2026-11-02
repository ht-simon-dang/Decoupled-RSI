#!/usr/bin/env python3
"""Standard-library checks for the local index and service lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


BACKEND = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data=json.dumps({'info':{'title':sys.argv[2]}}).encode()
        self.send_response(200)
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self,*args): pass
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


class HubChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.launcher = app.Launcher(data_dir=Path(self.temp.name))
        self.server = app.HubServer(("127.0.0.1", 0), self.launcher)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.launcher.close()
        self.temp.cleanup()

    def request(self, path, method="GET", headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def test_index_alias_assets_and_file_boundary(self):
        for path in ("/", "/index.php", "/index.html"):
            status, body, headers = self.request(path)
            self.assertEqual(status, 200)
            self.assertIn(b'Agent Workspace', body)
            self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(self.request('/shared.css')[0], 200)
        self.assertEqual(self.request('/../../app.py')[0], 404)
        self.assertEqual(self.request('/.local/runtime.env')[0], 404)

    def test_start_requires_local_origin_and_known_agent(self):
        with patch.object(self.launcher, 'start') as start:
            self.assertEqual(self.request('/api/agents/drug/start', 'POST', {'Origin':'https://example.org'})[0], 403)
            self.assertEqual(self.request('/api/agents/drug/start', 'POST', {'Sec-Fetch-Site':'cross-site'})[0], 403)
            self.assertEqual(self.request('/api/agents/unknown/start', 'POST')[0], 404)
            self.assertEqual(self.request('/', headers={'Host':f'example.org:{self.server.server_port}'})[0], 403)
            self.assertEqual(self.request('/api/agents/drug/start')[0], 404)
            start.assert_not_called()
            start.return_value = {'state':'running'}
            self.assertEqual(self.request('/api/agents/drug/start', 'POST')[0], 200)
            start.assert_called_once_with(app.BY_ID['drug'])

    def test_failure_is_reported_without_breaking_index(self):
        with patch.object(self.launcher, 'start', side_effect=RuntimeError('missing runtime')):
            status, body, _ = self.request('/api/agents/peptide/start', 'POST')
            self.assertEqual(status, 503)
            self.assertIn(b'missing runtime', body)
        self.assertEqual(self.request('/')[0], 200)

    def test_concurrent_alias_start_reuses_one_process_and_close_reaps_it(self):
        self.launcher.port_offset = free_port() - app.BY_ID['drug'].port
        real_popen = subprocess.Popen
        children = []

        def backend(command, **kwargs):
            agent = app.BY_ID[command[command.index('--worker') + 1]]
            port = command[command.index('--port') + 1]
            child = real_popen([sys.executable, '-c', BACKEND, port, agent.title], **kwargs)
            children.append(child)
            return child

        with patch('app.subprocess.Popen', side_effect=backend), patch('app.python_for', return_value=sys.executable):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(self.launcher.start, [app.BY_ID['drug'], app.BY_ID['self-evolve']]))
        self.assertEqual([r['state'] for r in results], ['running', 'running'])
        self.assertEqual(len(children), 1)
        self.assertEqual(results[1]['fragment'], 'evolve')
        self.launcher.close()
        self.assertIsNotNone(children[0].poll())

    def test_unrelated_port_is_never_reused_or_stopped(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            self.launcher.port_offset = occupied.getsockname()[1] - app.BY_ID['drug'].port
            with self.assertRaisesRegex(RuntimeError, '端口'):
                self.launcher.start(app.BY_ID['drug'])
            self.launcher.close()
            self.assertGreaterEqual(occupied.fileno(), 0)

    def test_matching_external_service_survives_hub_shutdown(self):
        self.launcher.data_dir = None
        port = free_port()
        self.launcher.port_offset = port - app.BY_ID['drug'].port
        process = subprocess.Popen([sys.executable, '-c', BACKEND, str(port), app.BY_ID['drug'].title])
        try:
            import time
            deadline = time.monotonic() + 5
            while self.launcher.probe(app.BY_ID['drug']) != 'running' and time.monotonic() < deadline:
                time.sleep(.05)
            status = self.launcher.start(app.BY_ID['drug'])
            self.assertFalse(status['managed'])
            self.launcher.close()
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_preview_does_not_reuse_external_data(self):
        with patch.object(self.launcher, 'probe', return_value='running'):
            with self.assertRaisesRegex(RuntimeError, '独立预览'):
                self.launcher.start(app.BY_ID['drug'])


if __name__ == '__main__':
    unittest.main()
