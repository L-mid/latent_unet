
import os, torch, pytest
from contextlib import nullcontext
from utils.debug import DebugManager

debug = DebugManager()

# === NOTES:
"""
Simple debugger test.
"""

# -------------- Helpers ---------------

@pytest.fixture(autouse=True)
def _reset_debug_state():
    """
    Reset the singleton between tests so state doesn't leak.
    """
    # hard reset before
    debug.disable()
    debug.reset()
    debug.log_timestamps = False
    debug.log_namespace = True
    yield
    # and after
    debug.disable()
    debug.reset()
    debug.log_timestamps = False
    debug.log_namespace = True



# ------------------- DebugManager basics ---------------------

def test_defaults_allow_all_namespaces():
    # with no tracked modules, everything is conidered tracked.
    assert debug.is_tracked("anything") is True


def test_enable_disable_toogle():
    debug.disable()
    assert not debug.enabled
    debug.enable()
    assert debug.enabled
    debug.toggle()
    assert not debug.enabled


def test_track_untrack_reset():
    debug.track("foo")
    assert debug.is_tracked("foo") is True
    assert debug.is_tracked("bar") is False     # once something is tracked, others aren't
    debug.untrack("foo")
    assert debug.is_tracked("foo") is True      # untracked & empty set => allow all again
    debug.track("x")
    debug.track("y")
    debug.reset()
    assert debug.is_tracked("anything") is True



# ------------------ debug_log behaviour ---------------------

def test_debug_log_respects_enabled_and_namespace(capfd):
    from utils.debug import debug, debug_log
    # When disabled: prints nothing
    debug.disable()
    debug_log("hello", name="ns")
    out, _ = capfd.readouterr()
    assert out == ""

    # When enabled + namespace on: prints with namespace
    debug.enable()
    debug.log_namespace = True
    debug_log("hello", name="ns")
    out, _ = capfd.readouterr()
    assert "[ns]" in out and "hello" in out

    # Namespace off: no [ns]
    debug.set_namespace(False)
    debug_log("world", name="ns")
    out, _ = capfd.readouterr()
    assert "[ns]" not in out and "world" in out


def test_debug_log_timesteps_flag(capfd):
    from utils.debug import debug, debug_log
    debug.enable()
    # flip timestampes directly (set_verbose has a known typo bug; see next test)
    debug.log_timestamps = True
    debug_log("tick", name="t")
    out, _ = capfd.readouterr()
    # crude check starts with [HH:MM:SS]
    assert out.startswith("[") and "]" in out.split(" ", 1)[0]


def test_set_verbose_should_toggle_timestamps(capfd):
    from utils.debug import debug, debug_log
    debug.enable()
    debug.set_verbose(True)
    debug_log("tick", name="t")
    out, _ = capfd.readouterr()
    assert out.startswith("[")


def test_debug_section_enter_exit(capfd):
    from utils.debug import debug, debug_section
    debug.enable()
    debug.track("zone")
    with debug_section("zone"):
        pass
    out, _ = capfd.readouterr()
    assert ">> Enter" in out and "<< Exit (" in out


# ----------- attach_debug_hooks_behavior (PyTorch) ------------

def test_attach_debug_hooks_noop_when_disabled(capfd):
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    from utils.debug import debug, attach_debug_hooks
    debug.disable()     # hooks should not be attached

    m = nn.Linear(4, 2)
    attach_debug_hooks(m, module_name="mylin")

    x = torch.randn(1, 4)
    _ = m(x)    # should produce no debug output
    out, _ = capfd.readouterr()
    assert out == ""

def test_attach_debug_hooks_logs_forward_shapes(capfd):
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    from utils.debug import debug, attach_debug_hooks
    debug.enable()

    m = nn.Linear(4, 2)
    attach_debug_hooks(m, module_name="mylin")

    x = torch.randn(1, 4)
    _ = m(x)

    out, _ = capfd.readouterr()
    # The current implemenation prints a single line with class + shapes
    assert "Forward: Linear" in out
    assert "Input shape:" in out
    assert "Output shape:" in out
    assert "name="




























