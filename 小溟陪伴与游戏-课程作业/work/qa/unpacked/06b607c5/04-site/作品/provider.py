"""DeepSeek transport and the original read-only desktop configuration."""
import base64
import json
import os
import urllib.error
import urllib.request

from companion import DEFAULT_PERSONALITY, _dpapi, default_home


class UserError(ValueError):
    pass


class Provider:
    def __init__(self, home, isolated=False):
        self.file = home / "settings.json"
        self.key = ""
        self.consent = False
        self.model = "deepseek-flash"
        self.personality = DEFAULT_PERSONALITY
        source = self.file
        if not source.exists() and not isolated:
            source = default_home() / "settings.json"
        if source.is_file():
            try:
                data = json.loads(source.read_text(encoding="utf-8-sig"))
                settings = data.get("settings", {})
                self.consent = settings.get("cloud_consent") is True
                self.personality = settings.get("personality_prompt") or DEFAULT_PERSONALITY
                if settings.get("deepseek_model") == "deepseek-v4-pro":
                    self.model = "deepseek-v4-pro"
                if os.name == "nt" and data.get("protected_keys"):
                    decoded = _dpapi(base64.b64decode(data["protected_keys"]), decrypt=True)
                    self.key = json.loads(decoded).get("deepseek", "")
            except (OSError, ValueError, TypeError, KeyError):
                self.key = ""

    def status(self):
        return {"configured": bool(self.key), "consent": self.consent,
                "cloud": bool(self.key and self.consent)}

    def update(self, key, consent):
        if not isinstance(key, str) or len(key) > 512 or any(c.isspace() for c in key):
            raise UserError("请检查密钥长度和空格。")
        chosen_key = key or self.key
        if consent and not chosen_key:
            raise UserError("请先填写 DeepSeek 密钥。")
        protected = None
        if chosen_key and os.name == "nt":
            protected = base64.b64encode(_dpapi(json.dumps({"deepseek": chosen_key}).encode())).decode()
        data = {"protected_keys": protected, "settings": {"cloud_consent": consent,
                "personality_prompt": self.personality, "deepseek_model": self.model}}
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file)
        self.key, self.consent = chosen_key, consent
        return self.status()

    def ask(self, messages, json_reply=False):
        if not self.status()["cloud"]:
            raise UserError("请先在连接设置中填写密钥并授权。")
        payload = {"model": self.model, "messages": messages, "max_tokens": 650,
                   "thinking": {"type": "disabled"}}
        if json_reply:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request("https://api.deepseek.com/chat/completions",
            json.dumps(payload).encode(), {"Content-Type": "application/json",
            "Authorization": "Bearer " + self.key}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=35) as result:
                text = json.load(result)["choices"][0]["message"]["content"].strip()
                if not text:
                    raise ValueError
                return json.loads(text) if json_reply else text
        except urllib.error.HTTPError as exc:
            errors = {401: "密钥无效", 403: "没有访问权限", 402: "账户余额不足", 429: "调用过于频繁"}
            raise UserError("DeepSeek：" + errors.get(exc.code, f"请求失败（{exc.code}）") + "。请检查设置后重试。") from None
        except (OSError, ValueError, KeyError, IndexError, AttributeError):
            raise UserError("DeepSeek 连接或响应异常，请稍后重试。") from None
