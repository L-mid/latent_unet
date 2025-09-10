
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional

# === NOTES:
"""
Manages fails.
"""

@dataclass
class Rule:
    mode: str           # "once" | "nth" | "always" | "rate"
    n: int = 1          # for nth
    count: int = 0
    p: float = 1.0      # for rate
    exc_factory: Callable[[], BaseException] = lambda: Exception("injected")


class Failpoints:
    def __init__(self):
        self.rules: Dict[str, Rule] = {}

    # APIs
    def once(self, key: str, exc: BaseException):
        self.rules[key] = Rule(mode="once", exc_factory=lambda e=exc: e)

    def nth(self, key: str, n: int, exc: BaseException):
        self.rules[key] = Rule(mode="nth", n=n, exc_factory=lambda e=exc: e)

    def always(self, key: str, n: int, exc: BaseException):
        self.rules[key] = Rule(mode="always", exc_factory=lambda e=exc: e)

    def should_raise(self, key: str) -> Optional[BaseException]:
        r = self.rules.get(key)
        if not r:
            return None
        if r.mode == "always":
            return r.exc_factory() 
        if r.mode == "once":
            del self.rules[key]
            return r.exc_factory()
        if r.mode == "nth":
            r.count += 1
            if r.count == r.n:
                # keep rule to be deterministic across multiple calls? remove:
                del self.rules[key]
                return r.exc_factory()
        return None
    

failpoints = Failpoints()

@contextmanager

def enable_failpoints():
    try:
        yield failpoints
    finally:
        failpoints.rules.clear()























