import ctypes, os
from pathlib import Path

DEFAULT_PERSONALITY = (
    "你叫小溟，是一个温和、真诚、有一点轻盈幽默感的 AI 桌面伙伴。"
    "像平等的聊天伙伴一样说话，少用套话和说教，先接住对方的具体处境。"
    "日常回复简短自然，不连续追问；对方想倾诉时认真听，想讨论办法时再一起拆解。"
    "不称用户为主人，不扮演恋人，不卖惨，不要求陪伴或回复。"
    "面对严肃或安全相关话题，收起玩笑，清楚、平静地提供支持。"
)

class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]

def _dpapi(data: bytes, *, decrypt: bool = False) -> bytes:
    buffer = ctypes.create_string_buffer(data)
    incoming = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = _Blob()
    if decrypt:
        ok = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)
        )
    else:
        ok = ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)
        )
    if not ok:
        raise OSError("Windows credential protection failed")
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        ctypes.windll.kernel32.LocalFree(outgoing.data)

def default_home() -> Path:
    override = os.environ.get("ECO_PET_HOME")
    if override:
        return Path(override).resolve()
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "EcoCompanion" / "pet"
