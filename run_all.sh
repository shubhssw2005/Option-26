#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Fully automatic: data collection + server, market-hours aware
#
# Usage:  ./run_all.sh
#
# What it does:
#   - Starts FastAPI server in background
#   - Detects market status (IST time)
#   - OPEN (09:15–15:30):  live snapshots every 60s
#   - POST-MARKET (15:30–18:00): EOD collection + model retrain
#   - CLOSED: uses previous day's data, sleeps until next open
#   - Loops forever — Ctrl+C to stop
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_DATA="/opt/anaconda3/bin/python3"
PYTHON_TRAIN="/opt/anaconda3/envs/torch_env/bin/python"
UVICORN="/opt/anaconda3/envs/torch_env/bin/uvicorn"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/logs"
mkdir -p "$LOG"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log()  { print -P "%B[$(TZ=Asia/Kolkata date '+%H:%M:%S IST')]%b $1"; }
ok()   { print -P "${GREEN}✓${NC} $1"; }
warn() { print -P "${YELLOW}⚠${NC}  $1"; }

ist_hhmm() { TZ="Asia/Kolkata" date '+%H:%M'; }
ist_dow()  { TZ="Asia/Kolkata" date '+%u'; }  # 1=Mon 7=Sun

market_status() {
    local t=$(ist_hhmm)
    local d=$(ist_dow)
    [[ $d -ge 6 ]] && echo "closed" && return
    [[ "$t" > "09:14" && "$t" < "15:31" ]] && echo "open"   && return
    [[ "$t" > "15:30" && "$t" < "18:01" ]] && echo "post"   && return
    echo "closed"
}

secs_until_open() {
    # Seconds until 09:15 IST next trading day
    local now_secs=$(TZ="Asia/Kolkata" date '+%s')
    local d=$(ist_dow)
    local add_days=1
    [[ $d -eq 5 ]] && add_days=3
    [[ $d -eq 6 ]] && add_days=2
    local open_secs=$(TZ="Asia/Kolkata" date -v+${add_days}d -v9H -v15M -v0S '+%s' 2>/dev/null || \
                      TZ="Asia/Kolkata" date -d "+${add_days} days 09:15:00" '+%s')
    echo $((open_secs - now_secs))
}

SERVER_PID=""

start_server() {
    log "Starting server on ${BLUE}http://localhost:8000${NC}"
    cd "$DIR"
    OMP_NUM_THREADS=1 "$UVICORN" vol_server:app --host 0.0.0.0 --port 8000 \
        >> "$LOG/server.log" 2>&1 &
    SERVER_PID=$!
    sleep 3
    if kill -0 $SERVER_PID 2>/dev/null; then
        ok "Server running (PID $SERVER_PID)"
        ok "Open: ${BLUE}http://localhost:8000${NC}"
    else
        warn "Server failed to start — check $LOG/server.log"
    fi
}

stop_server() {
    [[ -n "$SERVER_PID" ]] && kill $SERVER_PID 2>/dev/null && log "Server stopped"
}

collect_live() {
    log "Live collection (market OPEN)"
    cd "$DIR"
    # Run in foreground so we can detect when market closes
    "$PYTHON_DATA" collect_data.py --live 60
    # collect_data.py --live exits automatically at 15:30
}

collect_eod() {
    log "EOD collection (post-market)"
    cd "$DIR"
    "$PYTHON_DATA" collect_data.py --eod
    ok "EOD done"
}

retrain() {
    log "Retraining models with today's data..."
    cd "$DIR"
    OMP_NUM_THREADS=1 "$PYTHON_TRAIN" build_model.py NIFTY BANKNIFTY FINNIFTY MIDCPNIFTY \
        >> "$LOG/training.log" 2>&1
    ok "Models retrained"
}

db_stats() {
    local s=$(sqlite3 "$DIR/data.db" "SELECT COUNT(*) FROM option_chain_snapshot" 2>/dev/null || echo 0)
    local c=$(sqlite3 "$DIR/data.db" "SELECT COUNT(*) FROM historical_candle"     2>/dev/null || echo 0)
    local o=$(sqlite3 "$DIR/data.db" "SELECT COUNT(*) FROM historical_option"     2>/dev/null || echo 0)
    log "DB: snapshots=$s  candles=$c  opt_candles=$o"
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
trap 'log "Shutting down..."; stop_server; exit 0' INT TERM

# ── Main loop ─────────────────────────────────────────────────────────────────
print -P "\n${BOLD}Options Intelligence Platform — Auto Runner${NC}"
print -P "${BLUE}────────────────────────────────────────${NC}"
log "IST: $(TZ=Asia/Kolkata date '+%H:%M:%S %a %d %b %Y')"
log "Market: $(market_status | tr '[:lower:]' '[:upper:]')"
print -P "${BLUE}────────────────────────────────────────${NC}\n"

# Start server once
start_server

EOD_DONE=false   # track if we've done EOD for today

while true; do
    mkt=$(market_status)

    case "$mkt" in

        open)
            EOD_DONE=false
            log "Market OPEN — starting live collection"
            collect_live   # blocks until 15:30
            log "Market closed — switching to EOD"
            ;;

        post)
            if [[ "$EOD_DONE" == false ]]; then
                collect_eod
                db_stats
                retrain
                EOD_DONE=true
                log "EOD complete. Server still running with latest data."
            else
                log "EOD already done today. Sleeping 30 min..."
                sleep 1800
            fi
            ;;

        closed)
            local wait_secs=$(secs_until_open)
            local wait_min=$((wait_secs / 60))
            log "Market CLOSED — next open in ~${wait_min} min"
            log "Server running with last available data."
            db_stats

            # Sleep in 30-min chunks so we wake up near open
            if [[ $wait_secs -gt 1800 ]]; then
                sleep 1800
            else
                sleep $((wait_secs > 60 ? wait_secs - 60 : 60))
            fi
            EOD_DONE=false  # reset for next day
            ;;

    esac
done
