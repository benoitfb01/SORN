#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
artifact_root="$repo_root/artifacts/differential"
python2="$repo_root/artifacts/toolchains/miniconda2/bin/python"
reference_commit=cdad55d55f39e04f568ca1bc0c6036bec8db08fb

mkdir -p "$artifact_root/python2_source" "$artifact_root/results"
if [ ! -x "$python2" ]; then
  echo "Python 2 reference runtime not found: $python2" >&2
  echo "See DIFFERENTIAL_VALIDATION.md for the required environment." >&2
  exit 2
fi
git -C "$repo_root" cat-file -e "$reference_commit^{commit}"
git -C "$repo_root" archive "$reference_commit" | tar -x -C "$artifact_root/python2_source"

SORN_RUNTIME=python2 \
SORN_SOURCE_ROOT="$artifact_root/python2_source" \
arch -x86_64 "$python2" "$repo_root/differential/core_case.py" \
  "$artifact_root/results/python2.npz"

"$repo_root/.venv/bin/python" "$repo_root/differential/core_case.py" \
  "$artifact_root/results/python3.npz"

"$repo_root/.venv/bin/python" "$repo_root/differential/compare.py" \
  "$artifact_root/results/python2.npz" \
  "$artifact_root/results/python3.npz" \
  --report "$artifact_root/results/report.json"
