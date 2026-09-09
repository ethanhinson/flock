#!/usr/bin/env bash
# run.sh — start a 2-node swarm on this Mac, reachable from your phone.
#   ./run.sh          two nodes, 28 layers split 14/14
#   ./run.sh 3        three nodes
#   ./run.sh 2 remote add a second machine: see README
set -euo pipefail
cd "$(dirname "$0")"

# Find a python that has mlx-lm. uv tool installs land outside your PATH python,
# so check there too before giving up.
PY="${FLOCK_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in \
    "$(command -v python3)" \
    "$HOME/.local/share/uv/tools/mlx-lm/bin/python"; do
    if [ -x "$cand" ] && "$cand" -c "import mlx_lm" 2>/dev/null; then PY="$cand"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "error: no python with mlx-lm found."
  echo "  install it:  uv tool install --with flask --with requests --with numpy mlx-lm"
  echo "  or point at one:  FLOCK_PYTHON=/path/to/python ./run.sh"
  exit 1
fi
N=${1:-2}
LAYERS=28                      # Qwen3-1.7B
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)

cleanup() { echo; echo "stopping swarm…"; kill $(jobs -p) 2>/dev/null || true; }
trap cleanup EXIT INT TERM

SHARDS=()
per=$(( LAYERS / N ))
for i in $(seq 0 $((N-1))); do
  start=$(( i * per ))
  end=$(( i == N-1 ? LAYERS-1 : start + per - 1 ))
  port=$(( 8001 + i ))
  echo "node $i -> layers $start-$end on :$port"
  $PY shard.py --start $start --end $end --port $port >"/tmp/flock-shard$i.log" 2>&1 &
  SHARDS+=("$IP:$port")
done

echo "waiting for shards to load weights…"
for s in "${SHARDS[@]}"; do
  until curl -sf "http://$s/info" >/dev/null 2>&1; do sleep 1; done
  echo "  ✓ $s"
done

echo
echo "  ┌────────────────────────────────────────────┐"
echo "  │  open on your PHONE (same wifi):           │"
echo "  │    http://$IP:8000"
echo "  └────────────────────────────────────────────┘"
echo
$PY swarm.py --shards "${SHARDS[@]}" --port 8000
