"""eval/longmemeval.py leaves no benchmark data behind (owner, 2026-11-14), even when a run is
killed: the 2026-11-14 S run was killed with SIGKILL and left 232 MB of imported data."""

import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "longmemeval.py"
spec = importlib.util.spec_from_file_location("longmemeval_script", SCRIPT)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_killed_run_s_directory_is_removed_by_the_next_run(tmp_path: Path) -> None:
    killed = tmp_path / f"longmemeval-s-{dead_pid()}-abc"
    (killed / "dataset").mkdir(parents=True)
    alive = tmp_path / f"longmemeval-m-{os.getpid()}-def"  # a run still going: kept
    alive.mkdir()
    other = tmp_path / "longmemeval-notes"  # not a run directory: kept
    other.mkdir()
    assert script.remove_stale(tmp_path) == [killed]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([alive.name, other.name])


def test_a_terminated_run_still_deletes_its_directory(tmp_path: Path) -> None:
    """SIGTERM ends the run through its `finally`; the engine's databases live inside too."""
    code = f"""
import os, signal, sys, tempfile
sys.path.insert(0, {str(SCRIPT.parent)!r})
import longmemeval
work = longmemeval.workspace("s", {str(tmp_path)!r})
with work as path:
    db = tempfile.mkdtemp(prefix="lite-mem-eval-")  # as the engine's databases are made
    open(os.path.join(db, "x.db"), "w").close()
    print(path, db, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    signal.pause()
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 128 + signal.SIGTERM, proc.stderr
    work, db = map(Path, proc.stdout.split())
    assert work.name.startswith("longmemeval-s-") and db.parent == work
    assert list(tmp_path.iterdir()) == [] and not db.exists()
