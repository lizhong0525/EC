"""Run with Python 3.10+. Only public/ is served to the browser."""
import argparse
import json
import os
import secrets
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "vendor"))
import engine
from games import act
from provider import Provider, UserError


def demo_reply(text, history):
    if any(word in text for word in ("自杀", "不想活", "自残")):
        return "谢谢你说出来。你现在安全吗？如果你可能马上伤害自己，请先联系身边可信任的人并拨打当地急救电话。"
    if any(word in text for word in ("游戏", "棋", "卧底")):
        return "游戏间有两张桌子：国际象棋适合自己慢慢想，谁是卧底要仔细听大家说话。你更想试哪一个？"
    if any(word in text for word in ("紧张", "压力", "焦虑", "考试", "累")):
        return "听起来你已经绷紧了一阵子。如果愿意，我们可以先从今天最让你费神的那件小事说起。"
    if any(word in text for word in ("开心", "成功", "完成", "高兴")):
        return "听起来这是值得留住的一刻。你愿意讲讲，哪一部分最让你开心吗？"
    if any(word in text for word in ("你好", "早上好", "晚上好")):
        return "你好，我是小溟。我在这里，今天想随便聊聊，还是说说发生了什么？"
    if history:
        return "我还在听。你刚才提到的事情里，有没有哪一个细节特别想再说说？"
    return "我在听。你可以从最想说的那个片段开始，不用先把事情理顺。"


class App:
    def __init__(self, home, isolated):
        home.mkdir(parents=True, exist_ok=True)
        self.file = home / "state.json"
        self.provider = Provider(home, isolated)
        self.lock = threading.RLock()
        self.data = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {"chat": [], "games": {}}

    def save(self, data):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file)
        self.data = data

    def game_view(self, game):
        view = engine.view(game)
        view["mode"] = game["mode"]
        if game["kind"] == "chess" and game["mode"] == "local":
            view["pgn"] = view["pgn"].replace("DeepSeek", "Local rules")
        return view

    def snapshot(self):
        return {"settings": self.provider.status(), "chat": self.data["chat"],
                "games": {kind: self.game_view(g) for kind, g in self.data["games"].items()}}

    def request(self, route, form):
        import copy
        with self.lock:
            if route == "/api/state":
                return self.snapshot()
            if route == "/api/settings":
                return self.provider.update(form.get("key", ""), form.get("consent") is True)
            data = copy.deepcopy(self.data)
            mode = form.get("mode", "local")
            if route in ("/api/chat", "/api/game") and mode not in ("local", "cloud"):
                raise UserError("请选择可用模式。")
            if route in ("/api/chat", "/api/game") and mode == "cloud" and not self.provider.status()["cloud"]:
                raise UserError("请先配置并授权 DeepSeek。")
            if route == "/api/chat":
                text = form.get("text", "")
                token = form.get("request_id", "")
                if not isinstance(text, str) or not 0 < len(text.strip()) <= 1500:
                    raise UserError("请输入 1 至 1500 字。")
                if not isinstance(token, str) or not 1 <= len(token) <= 100:
                    raise UserError("缺少本轮消息编号。")
                if any(turn["id"] == token for turn in data["chat"]):
                    return data["chat"]
                history = [turn for turn in data["chat"] if turn["mode"] == mode]
                if mode == "cloud":
                    messages = [{"role": "system", "content": self.provider.personality}]
                    for turn in history[-6:]:
                        messages.extend([{"role":"user","content":turn["user"]}, {"role":"assistant","content":turn["answer"]}])
                    answer = self.provider.ask(messages + [{"role":"user","content":text.strip()}])
                else:
                    answer = demo_reply(text, history)
                data["chat"].append({"id": token, "user": text.strip(), "answer": answer, "mode": mode})
                result = data["chat"]
            elif route in ("/api/game", "/api/action"):
                kind = form.get("kind")
                if kind not in ("chess", "undercover"):
                    raise UserError("请选择游戏。")
                if route == "/api/game":
                    color = form.get("color", "white")
                    if color not in ("white", "black"):
                        raise UserError("请选择执棋颜色。")
                    game = engine.new_chess(color) if kind == "chess" else engine.new_undercover()
                    game.update(id=secrets.token_urlsafe(16), mode=mode)
                    data["games"][kind] = game
                else:
                    game = data["games"].get(kind)
                    if not game:
                        raise UserError("请先开始一局。")
                    act(game, form, self.provider)
                result = self.game_view(game)
            else:
                raise UserError("接口不存在。")
            self.save(data)
            return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "public"), **kwargs)

    def log_message(self, *args):
        pass

    def list_directory(self, path):
        self.send_error(404)

    def local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in allowed and (not origin or origin in {"http://" + h for h in allowed})

    def reply(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.local_request():
            return self.reply({"error":"请从本机首页访问。"}, 403)
        if urlsplit(self.path).path == "/api/state":
            return self.reply(self.server.app.request("/api/state", {}))
        return super().do_GET()

    def do_POST(self):
        if not self.local_request() or self.headers.get_content_type() != "application/json":
            return self.reply({"error":"请从本机首页提交。"}, 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 16000:
                raise UserError("请求过长或缺少内容。")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise UserError("请求格式无效。")
            return self.reply(self.server.app.request(urlsplit(self.path).path, body))
        except (ValueError, KeyError, TypeError) as exc:
            return self.reply({"error":str(exc) if isinstance(exc,(UserError,engine.RuleError)) else "请求内容有误。"}, 400)
        except (ConnectionError, OSError):
            return self.reply({"error":"本地保存失败，请检查磁盘后重试。"}, 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--isolated", action="store_true", help="Use a separate local test store; ignore desktop credentials")
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()
    if not args.profile.isascii() or not args.profile.replace("-","").isalnum():
        parser.error("profile must be an ASCII name")
    dest = ROOT / ".runtime" / ("test-" + args.profile if args.isolated else args.profile)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.app = App(dest, args.isolated)
        url = f"http://127.0.0.1:{server.server_port}/"
        print(url, flush=True)
        if not args.no_browser:
            threading.Timer(.3, lambda: webbrowser.open(url)).start()
        server.serve_forever()
