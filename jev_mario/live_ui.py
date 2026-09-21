"""Tiny read-only live monitor for a realtime run directory."""
import argparse
import http.server
import pathlib
import socketserver

PAGE = '''<!doctype html><meta charset="utf-8"><title>Jev Mario Live</title>
<style>body{font:16px system-ui;background:#111;color:#eee;margin:2em}main{display:grid;grid-template-columns:1fr 1fr;gap:1em;max-width:760px}.card{background:#222;padding:1em;border-radius:10px}b{font-size:2em;color:#7dd3fc}pre{white-space:pre-wrap}</style>
<h1>Jev × Mario</h1><div class=card><img id=game src="live_frame.png" style="image-rendering:pixelated;width:512px;max-width:100%;background:#000"></div><main><div class=card><div>Jev 선택</div><b id=jev>–</b></div><div class=card><div>Mario 적용 행동</div><b id=act>–</b></div><div class=card><div>월드 X</div><b id=x>–</b></div><div class=card><div>응답 지연</div><b id=lat>–</b></div><div class=card><div>점프 단계</div><b id=phase>–</b></div><div class=card><div>상태</div><b id=status>–</b></div></main><p id=meta></p><script>
async function tick(){try{let n=Date.now();let d=await (await fetch('live_state.json?'+n)).json(); document.querySelector('#game').src='live_frame.png?'+n; document.querySelector('#jev').textContent=d.jev_action||'대기';document.querySelector('#act').textContent=d.current_action||'NOOP';document.querySelector('#x').textContent=d.world_x??d.summary?.max_world_x??'–';document.querySelector('#lat').textContent=d.latency_s?d.latency_s.toFixed(3)+' s':'–';document.querySelector('#phase').textContent=d.jump_phase||'–';document.querySelector('#status').textContent=d.status||'running';document.querySelector('#meta').textContent=`decision ${d.decision??'–'} · frame ${d.frame??'–'} · stale ${d.stale_state_frames??'–'} frames`;}catch(e){document.querySelector('#status').textContent='연결 대기';}}tick();setInterval(tick,250);
</script>'''

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0] == '/live_frame.png':
            p = pathlib.Path(self.server.run_dir) / 'live_frame.png'
            if not p.exists(): self.send_response(404); self.end_headers(); return
            self.send_response(200); self.send_header('Content-Type','image/png'); self.end_headers(); self.wfile.write(p.read_bytes()); return
        if self.path.split('?')[0] == '/live_state.json':
            p = pathlib.Path(self.server.run_dir) / 'live_state.json'
            data = p.read_bytes() if p.exists() else b'{"status":"starting"}'
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data); return
        body = PAGE.encode()
        self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run_dir'); ap.add_argument('--port', type=int, default=8765); a=ap.parse_args()
    with socketserver.TCPServer(('127.0.0.1', a.port), Handler) as s:
        s.run_dir = str(pathlib.Path(a.run_dir).resolve()); print(f'http://127.0.0.1:{a.port}/', flush=True); s.serve_forever()
if __name__ == '__main__': main()
