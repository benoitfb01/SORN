#!/usr/bin/env bash
set -euo pipefail

# Kubernetes Indexed Jobs number their pods from zero.  Derive stable values
# here so a retried index reuses the same seed and output directory.  Explicit
# SORN_RUN_ID/SORN_SEED values still take precedence for Docker or Slurm runs.
if [[ -n "${JOB_COMPLETION_INDEX:-}" ]]; then
  case "$JOB_COMPLETION_INDEX" in
    ''|*[!0-9]*)
      echo "JOB_COMPLETION_INDEX must be a non-negative integer" >&2
      exit 64
      ;;
  esac

  run_number=$((JOB_COMPLETION_INDEX + 1))
  if [[ -z "${SORN_RUN_ID:-}" ]]; then
    SORN_RUN_ID=$(printf 'n200-run-%03d' "$run_number")
  fi
  if [[ -z "${SORN_SEED:-}" ]]; then
    SORN_SEED=$((200000 + run_number))
  fi
fi

: "${SORN_RUN_ID:?Set SORN_RUN_ID, or run this as a Kubernetes Indexed Job}"
: "${SORN_SEED:?Set SORN_SEED, or run this as a Kubernetes Indexed Job}"

case "$SORN_RUN_ID" in
  *[!A-Za-z0-9._-]*)
    echo "SORN_RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 64
    ;;
esac

case "$SORN_SEED" in
  ''|*[!0-9]*)
    echo "SORN_SEED must be a non-negative integer" >&2
    exit 64
    ;;
esac

mkdir -p "/opt/sorn/backup/$SORN_RUN_ID"
export SORN_RUN_ID SORN_SEED

echo "Starting original SORN commit cdad55d55f39e04f568ca1bc0c6036bec8db08fb"
echo "Experiment: delpapa.param_Zheng2013; N_e=200; steps=5000000"
echo "Run ID: $SORN_RUN_ID; seed: $SORN_SEED"

exec /opt/conda/bin/python -u test_single.py delpapa.param_Zheng2013
