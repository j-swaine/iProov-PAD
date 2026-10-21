"""
Run a queue of jobs on parallel Colab GPU sessions, unattended.
================================================================

    python scripts/orchestrate.py QUEUE.yaml [--slots 4] [--gpu L4] [--poll 120]

QUEUE.yaml lists jobs (see configs/runs/sweep.yaml):

    pull_data: true          # pull the crop + synthetic caches onto each VM (false for smoke tests)
    jobs:
      - name: A_dinov2b_loose224
        config: configs/runs/A_dinov2b_loose224.yaml   # passed to `screenpad train --config ... --push`
        set: [seed=43]                                 # optional extra --set overrides
      - name: smoke1
        smoke: 60                                      # or: the stand-in job, sleeping this many seconds
      - name: patchcore_loose224
        cmd: [screenpad, oneclass, --members, patchcore, --run-name, patchcore_loose224, --push]   # or: any command following the STATUS/push contract

Each slot is one Colab session (pad-q1, pad-q2, ...), run by its own thread:

1. `colab new --gpu L4`. If the account refuses (session limit or capacity), the slot retries every `--retry-every` seconds while jobs remain; the other slots carry the queue meanwhile.
2. Push the code tarball and HF token (`scripts/colab_job.sh push`), then pull the crop and synthetic caches from HF in the background on the VM.
3. Take the next pending job, start it in the background on the VM (it outlives the CLI's exec call), and poll its STATUS file.
4. On "done", independently confirm the run on HF: MANIFEST.json is downloaded and every file it lists must exist on the Hub with the right size. Only then is the job marked done.
5. Repeat until the queue is empty, then `colab stop`. A session is always stopped when its thread exits, whatever happened.

A failed job is retried once (on any slot). A session that stops answering is stopped and replaced, and its job goes back to the queue; runs resume from their last epoch only if the same VM survives, otherwise they restart.

Every Colab CLI call goes through one lock: the CLI hung when two sessions ran `colab exec` at the same moment (smoke test, 2026-10-18), so calls are serialised. They are short (a poll is a few seconds); anything long runs in the background on the VM and is polled. Every call also has a hard timeout.

State: results/orchestrator/<queue name>/state.json (every job's status, slot, attempts, timings) and events.log (every session and job event, timestamped). Both are rewritten on every change, so a crash of this script loses nothing; re-running it skips jobs already done.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MARKER = "__SCREENPAD_" + "STEP_OK__"
RUNS_REPO = "Jamie1701/screenpad-runs"
MAX_ATTEMPTS = 2
REMOTE = "/content/screenpad"  # code folder on the VM; the kernel's working directory does not persist between execs
CLI_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Queue:
    def __init__(self, path: Path, out_dir: Path):
        spec = yaml.safe_load(path.read_text())
        self.pull_data = bool(spec.get("pull_data", True))
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = out_dir / "state.json"
        self.lock = threading.Lock()
        old = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.jobs = {}
        for j in spec["jobs"]:
            prev = old.get(j["name"], {})
            done = prev.get("status") == "done"
            self.jobs[j["name"]] = {**j, "status": "done" if done else "pending", "attempts": prev.get("attempts", 0) if done else 0,
                                     "slot": prev.get("slot"), "started": prev.get("started"), "finished": prev.get("finished"), "note": prev.get("note")}
        self.save()

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.jobs, indent=2))
        tmp.replace(self.state_path)

    def event(self, slot: str, msg: str) -> None:
        line = f"{now()} [{slot}] {msg}"
        print(line, flush=True)
        with open(self.out / "events.log", "a") as fh:
            fh.write(line + "\n")

    def take(self, slot: str) -> dict | None:
        with self.lock:
            for j in self.jobs.values():
                if j["status"] == "pending":
                    j.update(status="running", slot=slot, started=now(), attempts=j["attempts"] + 1)
                    self.save()
                    return dict(j)
        return None

    def finish(self, name: str, ok: bool, note: str = "") -> None:
        with self.lock:
            j = self.jobs[name]
            if ok:
                j.update(status="done", finished=now(), note=note)
            else:
                j.update(status="pending" if j["attempts"] < MAX_ATTEMPTS else "failed", note=note)
            self.save()

    def pending(self) -> int:
        with self.lock:
            return sum(j["status"] == "pending" for j in self.jobs.values())

    def remaining(self) -> int:
        with self.lock:
            return sum(j["status"] in ("pending", "running") for j in self.jobs.values())


class Session:
    def __init__(self, name: str, gpu: str, q: Queue):
        self.name, self.gpu, self.q = name, gpu, q
        self.env = {**os.environ, "COLAB_SESSION": name}

    def sh(self, args: list[str], timeout: float, input: str | None = None) -> tuple[bool, str]:
        with CLI_LOCK:
            try:
                r = subprocess.run(args, input=input, capture_output=True, text=True, timeout=timeout, env=self.env, cwd=ROOT)
                return r.returncode == 0, r.stdout + r.stderr
            except subprocess.TimeoutExpired:
                return False, "local timeout"

    def py(self, code: str, timeout: float = 120) -> tuple[bool, str]:
        """Run Python in the VM kernel, from the code folder; succeeds only if the code printed the marker."""
        code = f"import os\nos.chdir({REMOTE!r})\n" + code
        ok, out = self.sh(["colab", "exec", "-s", self.name, "--timeout", str(timeout)], timeout + 60, input=code)
        return ok and MARKER in out, out

    def start(self) -> bool:
        ok, out = self.sh(["colab", "new", "-s", self.name, "--gpu", self.gpu], 600)
        if not ok:
            self.q.event(self.name, f"session refused: {out.strip()[-300:]}")
            return False
        self.q.event(self.name, f"session started ({self.gpu})")
        ok, out = self.sh(["bash", "scripts/colab_job.sh", "push"], 600)
        if not ok or MARKER not in out:
            self.q.event(self.name, f"bootstrap failed: {out.strip()[-500:]}")
            return False
        pull = ["screenpad", "pull-data", "--cache"] if self.q.pull_data else ["true"]
        ok, out = self.py(f"""
import subprocess
g = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", g.stdout.strip())
subprocess.run(["nvidia-smi"], stdout=open("/content/nvidia_smi_start.txt", "w"))
p = subprocess.Popen(["bash", "-c", {" ".join(pull) + " > /content/pull.log 2>&1; echo $? > /content/pull.rc"!r}], cwd={REMOTE!r}, start_new_session=True)
print({MARKER[:12]!r} + {MARKER[12:]!r})
""")
        if not ok:
            self.q.event(self.name, f"could not start data pull: {out.strip()[-500:]}")
            return False
        gpu = next((l for l in out.splitlines() if l.startswith("GPU:")), "GPU: ?")
        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(20)
            ok, out = self.py(f"""
import os
print("RC=" + (open("/content/pull.rc").read().strip() if os.path.exists("/content/pull.rc") else "running"))
if os.path.exists("/content/pull.rc"): print(open("/content/pull.log").read()[-800:])
print({MARKER[:12]!r} + {MARKER[12:]!r})
""", timeout=60)
            rc = next((l[3:] for l in out.splitlines() if l.startswith("RC=")), "running") if ok else "running"
            if rc == "0":
                self.q.event(self.name, ("data pulled; " if self.q.pull_data else "no data pull; ") + gpu)
                return True
            if rc != "running":
                self.q.event(self.name, f"data pull failed (rc {rc}): {out.strip()[-600:]}")
                return False
        self.q.event(self.name, "data pull timed out")
        return False

    def stop(self) -> None:
        ok, out = self.sh(["colab", "stop", "-s", self.name], 300)
        self.q.event(self.name, "session stopped" if ok else f"stop FAILED, check manually: {out.strip()[-300:]}")

    def launched(self, name: str) -> bool:
        """Did a launch reach the VM? (Its exec reply can be lost while the job is in fact running.)"""
        ok, out = self.py(f"""
print("PID=" + str(os.path.exists("results/runs/{name}.pid")))
print({MARKER[:12]!r} + {MARKER[12:]!r})
""", timeout=60)
        return ok and "PID=True" in out

    def launch(self, job: dict) -> bool:
        cmd = (["python", "-m", "screenpad.train.smoke_job", job["name"], str(job["smoke"])] if "smoke" in job
               else job["cmd"] if "cmd" in job  # any command that writes results/runs/<name>/STATUS and pushes itself
               else ["screenpad", "train", "--config", job["config"], "--set", f"train.name={job['name']}",
                     *[x for kv in job.get("set", []) for x in ("--set", kv)], "--push"])
        ok, out = self.py(f"""
import subprocess, os
os.makedirs("results/runs/{job['name']}", exist_ok=True)
for p in ("results/runs/{job['name']}/STATUS", "results/runs/{job['name']}.pid"):
    if os.path.exists(p): os.remove(p)
log = open("results/runs/{job['name']}.stdout", "a")
proc = subprocess.Popen({cmd!r}, cwd={REMOTE!r}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open("results/runs/{job['name']}.pid", "w").write(str(proc.pid))
print({MARKER[:12]!r} + {MARKER[12:]!r})
""")
        return ok

    def status(self, name: str) -> dict | None:
        """The job's STATUS file; a job whose process has exited without a final state counts as failed (with its output tail)."""
        ok, out = self.py(f"""
import json, os
p = "results/runs/{name}/STATUS"
st = json.loads(open(p).read()) if os.path.exists(p) else {{"state": "launching"}}
pid = int(open("results/runs/{name}.pid").read())
try:
    alive = open(f"/proc/{{pid}}/stat").read().split(")")[-1].split()[0] != "Z"  # an exited child stays a zombie until reaped
except FileNotFoundError:
    alive = False
if st.get("state") not in ("done", "failed") and not alive:
    tail = open("results/runs/{name}.stdout").read()[-2000:]
    st = {{"state": "failed", "error": "process exited without a final status: " + tail}}
print("STATUS=" + json.dumps(st))
print({MARKER[:12]!r} + {MARKER[12:]!r})
""", timeout=60)
        if not ok:
            return None
        line = next((l for l in out.splitlines() if l.startswith("STATUS=")), None)
        return json.loads(line[7:]) if line else None


def verify_on_hub(name: str) -> list[str]:
    """Independent check from this machine: every file in the run's MANIFEST.json exists on the Hub with its recorded size."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    try:
        manifest = json.loads(Path(hf_hub_download(RUNS_REPO, f"runs/{name}/MANIFEST.json", repo_type="model", force_download=True)).read_text())
    except Exception as e:  # noqa: BLE001
        return [f"no manifest: {e}"]
    sizes = {i.path: i.size for i in api.list_repo_tree(RUNS_REPO, path_in_repo=f"runs/{name}", recursive=True) if hasattr(i, "size")}
    return [f"{rel}: missing or wrong size" for rel, m in manifest.items() if sizes.get(f"runs/{name}/{rel}") != m["bytes"]]


def worker(slot: str, gpu: str, q: Queue, poll: float, max_quiet: float, retry_every: float, max_stall: float = 0) -> None:
    s = Session(slot, gpu, q)
    current = None
    try:
        while not s.start():
            s.stop()
            if q.pending() == 0:
                return
            q.event(slot, f"retrying in {retry_every:.0f}s")
            time.sleep(retry_every)
        launch_failures = 0
        while (job := q.take(slot)) is not None:
            current = job["name"]
            q.event(slot, f"job {job['name']} start (attempt {job['attempts']})")
            started = s.launch(job)
            for _ in range(2):  # a lost reply is not a failed launch: look for the job's pid before relaunching
                if started:
                    break
                q.event(slot, f"job {job['name']} launch got no reply; checking the VM")
                started = s.launched(job["name"]) or s.launch(job)
            if not started:
                q.event(slot, f"job {job['name']} launch failed")
                q.finish(job["name"], False, "launch failed")
                launch_failures += 1
                if launch_failures >= 2:  # this session is not taking jobs: replace it rather than burn the queue
                    return
                continue
            launch_failures = 0
            last_ok, last_epoch = time.time(), None
            last_change, last_st = time.time(), None
            while True:
                time.sleep(poll)
                st = s.status(job["name"])
                if st is None:
                    if time.time() - last_ok > max_quiet:
                        q.event(slot, f"session unresponsive for {max_quiet:.0f}s; giving job {job['name']} back")
                        q.finish(job["name"], False, "session lost")
                        return
                    continue
                last_ok = time.time()
                if st != last_st:
                    last_change, last_st = time.time(), st
                elif max_stall and st.get("state") not in ("done", "failed") and time.time() - last_change > max_stall:
                    # answering but not progressing: on an expensive GPU, fail it and release the session rather than wait
                    q.event(slot, f"job {job['name']} stalled: STATUS unchanged for {max_stall:.0f}s ({st}); failing it and stopping the session")
                    q.finish(job["name"], False, "stalled")
                    return
                if st.get("epoch") != last_epoch and st.get("state") == "running":
                    last_epoch = st.get("epoch")
                    q.event(slot, f"job {job['name']} epoch {last_epoch}/{st.get('epochs')}")
                if st.get("state") == "done":
                    problems = verify_on_hub(job["name"])
                    if problems:
                        q.event(slot, f"job {job['name']} finished but HF verification failed: {problems[:3]}")
                        q.finish(job["name"], False, "hub verification failed")
                    else:
                        q.event(slot, f"job {job['name']} done and verified on HF")
                        q.finish(job["name"], True)
                    break
                if st.get("state") == "failed":
                    q.event(slot, f"job {job['name']} failed: {str(st.get('error'))[-400:]}")
                    q.finish(job["name"], False, "failed on VM")
                    break
        current = None
    except Exception as e:  # noqa: BLE001  never leave a job marked running, never leave a session up
        q.event(slot, f"slot error: {e!r}")
    finally:
        if current is not None and q.jobs[current]["status"] == "running":
            q.finish(current, False, "slot lost")
        s.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", type=Path)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--poll", type=float, default=120)
    ap.add_argument("--max-quiet", type=float, default=1800, help="seconds without a status answer before a session is given up")
    ap.add_argument("--retry-every", type=float, default=600, help="seconds between attempts to open a refused session")
    ap.add_argument("--max-stall", type=float, default=0, help="seconds a job's STATUS may stay unchanged before it is failed and its session stopped (0 = off; jobs must report progress)")
    ap.add_argument("--prefix", default="pad-q", help="session name prefix, so two queues (e.g. L4 and A100) can run at once")
    a = ap.parse_args()
    q = Queue(a.queue, ROOT / "results" / "orchestrator" / a.queue.stem)
    q.event("main", f"queue {a.queue} with {q.remaining()} jobs to do, {a.slots} slots of {a.gpu}")
    threads: dict[str, threading.Thread] = {}
    respawns = 0
    while True:  # keep `slots` workers alive while jobs are pending; a slot that exits is replaced (bounded)
        for k in range(a.slots):
            name = f"{a.prefix}{k + 1}"
            t = threads.get(name)
            if (t is None or not t.is_alive()) and q.pending() and (t is None or respawns < 12):
                respawns += t is not None
                threads[name] = threading.Thread(target=worker, args=(name, a.gpu, q, a.poll, a.max_quiet, a.retry_every, a.max_stall), daemon=False)
                threads[name].start()
                time.sleep(20)  # stagger session creation
        if not any(t.is_alive() for t in threads.values()) and (not q.pending() or respawns >= 12):
            break
        time.sleep(30)
    summary = {s: sum(j["status"] == s for j in q.jobs.values()) for s in ("done", "failed", "pending")}
    q.event("main", f"finished: {summary}")
    sys.exit(0 if summary["failed"] == 0 and summary["pending"] == 0 else 1)


if __name__ == "__main__":
    main()
