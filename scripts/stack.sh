#!/usr/bin/env bash
# Start/stop the native Mac app. Notes run in a local MLX process, not Ollama.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.run"
APP_PORT="${APP_PORT:-8780}"
LLM_MODEL="${LLM_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  LLM_MODEL="${LLM_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

mkdir -p "$RUN"

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  local file="$1"
  [[ -f "$file" ]] && cat "$file" || true
}

stop_pidfile() {
  local file="$1"
  local pid
  pid="$(read_pid "$file")"
  if pid_alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      pid_alive "$pid" || break
      sleep 0.3
    done
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

cmd="${1:-}"
case "$cmd" in
  start)
    if [[ ! -x "$ROOT/.venv-app/bin/python" ]]; then
      echo "Missing app venv — run: make app-venv" >&2
      exit 1
    fi

    if ! pid_alive "$(read_pid "$RUN/app.pid")"; then
      echo "Starting app on :$APP_PORT (notes: ${LLM_MODEL}, offline hub) …"
      nohup env \
        DATA_DIR="$ROOT/data" \
        LLM_MODEL="$LLM_MODEL" \
        HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
        TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
        STENOPOD_ROOT="$ROOT" \
        "$ROOT/.venv-app/bin/python" -m uvicorn stenopod.main:app \
          --app-dir "$ROOT/app" \
          --host 127.0.0.1 \
          --port "$APP_PORT" \
        >"$RUN/app.log" 2>&1 &
      echo $! >"$RUN/app.pid"
    else
      echo "app already running (pid $(read_pid "$RUN/app.pid"))"
    fi
    echo "Open http://127.0.0.1:${APP_PORT}/"
    echo "Start capture from the UI (or: make sidecar / make sidecar-meet)."
    ;;
  stop)
    if [[ -x "$ROOT/.venv-app/bin/python" ]]; then
      curl -sf -X POST "http://127.0.0.1:${APP_PORT}/api/capture/stop" >/dev/null 2>&1 || true
    fi
    stop_pidfile "$RUN/app.pid"
    stop_pidfile "$RUN/sidecar.pid"
    echo "Stopped."
    ;;
  status)
    echo "notes: ${LLM_MODEL} (local MLX, HF_HUB_OFFLINE=${HF_HUB_OFFLINE})"
    for name in app sidecar; do
      pid="$(read_pid "$RUN/$name.pid")"
      if pid_alive "$pid"; then
        echo "$name: running pid $pid"
      else
        echo "$name: stopped"
      fi
    done
    ;;
  logs)
    files=("$RUN/app.log")
    [[ -f "$RUN/sidecar.log" ]] && files+=("$RUN/sidecar.log")
    tail -f "${files[@]}"
    ;;
  *)
    echo "Usage: $0 start|stop|status|logs" >&2
    exit 1
    ;;
esac
