import sys
import types
from types import SimpleNamespace


if "groq" not in sys.modules:
    groq = types.ModuleType("groq")
    groq.Groq = object
    sys.modules["groq"] = groq


def groq_response(content):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])
