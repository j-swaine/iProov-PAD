#!/usr/bin/env bash
# Drive a Colab GPU runtime from the local terminal with the Colab CLI.
#
#   scripts/colab_job.sh up [GPU]          start a session (default T4; "cpu" for none) and push the code
#   scripts/colab_job.sh push              re-send the current working tree and reinstall (same session)
#   scripts/colab_job.sh run ARGS...       run `screenpad ARGS...` on the VM, streaming its output
#   scripts/colab_job.sh py FILE.py        run a local Python file in the VM kernel
#   scripts/colab_job.sh get REMOTE LOCAL  download a file from the VM (REMOTE is relative to /, e.g. content/screenpad/results/x)
#   scripts/colab_job.sh down              stop the session (stops compute-unit billing)
#   scripts/colab_job.sh smoke [GPU]       up -> pull-data -> splits -> audit -> down
#
# The code goes up as a tarball of the working tree (tracked + untracked, minus .gitignore), so no
# GitHub token is needed on the VM and uncommitted edits are included. SOURCE_VERSION inside it
# records the commit and whether the tree was dirty. The HF token is uploaded as a file from the
# local `hf auth login`, not passed through code, so it never appears in the session history.
#
# `colab exec` exits 0 even when the remote code raises, so every remote snippet prints a marker as
# its last line and `checked` fails the step when the marker is missing.
set -euo pipefail

SESSION="${COLAB_SESSION:-screenpad}"
TIMEOUT="${COLAB_TIMEOUT:-86400}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
MARKER="__SCREENPAD_STEP_OK__"

checked() {
    # Run `colab exec ...` (args passed through), echo its output, fail if the marker is missing.
    local out
    out="$(mktemp)"
    colab exec -s "$SESSION" "$@" 2>&1 | tee "$out"
    if ! grep -q "$MARKER" "$out"; then
        rm -f "$out"
        echo "[colab_job] remote step failed" >&2
        return 1
    fi
    rm -f "$out"
}

up() {
    local gpu="${1:-T4}"
    if [[ "$gpu" == "cpu" ]]; then colab new -s "$SESSION"; else colab new -s "$SESSION" --gpu "$gpu"; fi
    push
}

push() {
    local tmp version
    tmp="$(mktemp -d)"
    version="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "no-commit")"
    [[ -n "$(git -C "$ROOT" status --porcelain)" ]] && version="$version-dirty"
    echo "$version" > "$tmp/SOURCE_VERSION"
    # bsdtar applies -C before reading -T, so append SOURCE_VERSION in a second pass, then compress.
    export COPYFILE_DISABLE=1
    (cd "$ROOT" && git ls-files -co --exclude-standard -z | tar --no-mac-metadata --null -cf "$tmp/screenpad.tar" -T -)
    tar --no-mac-metadata -rf "$tmp/screenpad.tar" -C "$tmp" SOURCE_VERSION
    gzip "$tmp/screenpad.tar"
    # Upload paths are relative to the VM's filesystem root, not /content.
    colab upload -s "$SESSION" "$tmp/screenpad.tar.gz" content/screenpad.tar.gz
    colab upload -s "$SESSION" "$HF_TOKEN_FILE" content/.hf_token
    rm -rf "$tmp"
    checked --timeout 900 -f "$ROOT/scripts/colab_bootstrap.py"
}

run() {
    # Stream the subprocess line by line: a plain subprocess.run would write to the kernel's own
    # stdout, which the CLI never sees.
    local args_py
    args_py="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"
    checked --timeout "$TIMEOUT" <<EOF
import subprocess
p = subprocess.Popen(["screenpad", *$args_py], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
if p.wait() != 0:
    raise SystemExit(f"screenpad exited with {p.returncode}")
print("__SCREENPAD_" + "STEP_OK__")
EOF
}

py() {
    # The file runs as-is, so it must print the marker itself on success (see colab_bootstrap.py).
    checked --timeout "$TIMEOUT" -f "$1"
}
get() { colab download -s "$SESSION" "$1" "$2"; }
down() { colab stop -s "$SESSION"; }

smoke() {
    up "${1:-T4}"
    trap down EXIT
    run pull-data
    run splits
    run audit
}

cmd="${1:-}"; shift || true
case "$cmd" in
    up|push|run|py|get|down|smoke) "$cmd" "$@" ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac
