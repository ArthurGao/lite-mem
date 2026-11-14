"""LongMemEval session recall for lite-mem, on lite-eval (design §8, §12 S11).

    uv run python eval/longmemeval.py s|m|oracle [--limit N] [--config NAME] [--concurrency 4]

Every run makes a temporary directory, downloads the official file there from Hugging Face
(`xiaowu0162/longmemeval-cleaned`, MIT), imports it with `lite-eval dataset import` (item by
item: M is 2.7 GB), runs the suite and deletes the directory, whatever happened (owner,
2026-11-14: benchmark data is never kept locally or committed): SIGTERM and SIGHUP end the run
through its cleanup, and a directory left by a run killed with SIGKILL is removed by the next.
Run records stay, in `eval/lite-eval/runs/longmemeval-<variant>/`; they hold session ids, not
benchmark text.

Each question is one user's history: its user turns become memories (`infer=False`, tagged with
their session), and the question's hits become session ids, so `recall@k` / `hit@k` are session
recall over the first k distinct sessions (`hit@k`: any evidence session; abstention questions
are skipped, as in the benchmark's own retrieval evaluation). bge-m3 on the local Ollama must be
running. Configs: `k10` and `k10-follow1` (top 10 memories, without and with the follow-up
round) and `k50` (top 50 memories: room for 10 distinct sessions).
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import httpx2
from lite_eval.datasets.loaders import import_dataset
from lite_eval.report.terminal import render_run
from lite_eval.runner import RunResult, run_suite
from lite_eval.store import load_run

ROOT = Path(__file__).resolve().parent / "lite-eval"
URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{}"
FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
SCORERS = ["hit@5", "recall@5", "hit@10", "recall@10"]
CONFIGS = [
    {"name": "k10", "top_k": 10},
    {"name": "k10-follow1", "top_k": 10, "follow_up": 1},
    {"name": "k50", "top_k": 50},
]


def download(name: str, to: Path) -> Path:
    path = to / name
    started, done = time.monotonic(), 0
    with httpx2.stream("GET", URL.format(name), follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
    print(f"downloaded {name}: {done:,} bytes in {time.monotonic() - started:.0f}s", flush=True)
    return path


def write_suite(variant: str, dataset: Path, to: Path, concurrency: int) -> Path:
    suite = {
        "name": f"longmemeval-{variant}",
        "dataset": str(dataset),
        "scorers": SCORERS,
        "concurrency": concurrency,
        "timeout_s": 3600,  # one M question embeds thousands of turns on a cold cache
        "targets": [{"python": "lite_mem.eval_target:longmemeval", "configs": CONFIGS}],
    }
    path = to / "suites" / f"longmemeval-{variant}.yaml"
    path.parent.mkdir()
    path.write_text(json.dumps(suite, indent=1))  # YAML is a superset of JSON
    return path


def report(result: RunResult) -> None:
    """lite-eval's run summary (errors, warnings, per question type), then what was stored:
    user turns longer than a memory may be are left out (the engine refuses them)."""
    print(render_run(result.record, str(result.directory)))
    spans = [s for o in load_run(result.directory).outputs for s in o.trace.spans]
    turns = sum(int(str(s.attrs["turns"])) for s in spans)
    stored = sum(int(str(s.attrs["stored"])) for s in spans)
    print(f"  user turns: {turns:,}, stored {stored:,}, left out {turns - stored:,}")


_RUN_DIR = re.compile(r"^longmemeval-[a-z]+-([0-9]+)-")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process
    return True


def remove_stale(parent: Path) -> list[Path]:
    """Run directories whose process is gone: a run killed with SIGKILL never reaches its
    `finally` (2026-11-14: an S run left 232 MB of imported data behind)."""
    removed = []
    for path in sorted(parent.iterdir()):
        m = _RUN_DIR.match(path.name)
        if m and path.is_dir() and not _alive(int(m.group(1))):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


def _stop(signum: int, frame: object) -> None:
    sys.exit(128 + signum)  # through `finally`, so the directory is deleted


@contextlib.contextmanager
def workspace(variant: str, parent: str | None = None) -> Iterator[Path]:
    """A directory for one run, deleted when it ends, however it ends short of SIGKILL (and
    after SIGKILL by the next run: `remove_stale`). The engine's temporary databases are made
    inside it too."""
    root = Path(parent or tempfile.gettempdir())
    for path in remove_stale(root):
        print(f"removed {path} (left by a run that was killed)", file=sys.stderr)
    work = Path(tempfile.mkdtemp(prefix=f"longmemeval-{variant}-{os.getpid()}-", dir=root))
    handlers = {s: signal.signal(s, _stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    previous, tempfile.tempdir = tempfile.tempdir, str(work)
    try:
        yield work
    finally:
        tempfile.tempdir = previous
        for s, handler in handlers.items():
            signal.signal(s, handler)
        shutil.rmtree(work, ignore_errors=True)
        print(f"removed {work}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(FILES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    with workspace(args.variant) as work:
        src = download(FILES[args.variant], work)
        started = time.monotonic()
        dataset = import_dataset("longmemeval", src, work / "dataset", limit=args.limit)
        src.unlink()  # the import holds what the run needs
        print(f"imported in {time.monotonic() - started:.0f}s", flush=True)
        suite = write_suite(args.variant, dataset, work, args.concurrency)
        started = time.monotonic()
        results = run_suite(suite, config=args.config, root=ROOT, no_cache=True)
        print(f"ran in {time.monotonic() - started:.0f}s", flush=True)
        for result in results:
            report(result)


if __name__ == "__main__":
    main()
