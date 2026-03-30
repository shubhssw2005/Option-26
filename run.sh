#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Smart launcher for the Options Intelligence Platform
#
# Usage:
#   ./run.sh              → auto-detect what to do based on IST time
#   ./run.sh server       → start FastAPI server only
#   ./run.sh collect      → collect data (auto: live/eod/historical by time)
#   ./run.sh train        → train all ML models
#   ./run.sh all          → collect + train + server
#   ./run.sh schedule     → run as a daily scheduler (keeps running)
#   ./run.sh mkt_status       → show current market mkt_status and DB stats
# ─────────────────────────────────────────────────────────────────────────────

set -e

PYTHON_DATA="/opt/anaconda3/bin/python3"          # base env — Nubra SDK auth
PYTHON_TRAIN="/opt/anaconda3/envs/torch_env/bin/python"  # torch env — ML training
UVICORN="/opt/anaconda3/envs/torch_env/bin/uvicorn"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BOLD}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
err()  { echo -e "${RED}✗${NC} $1"; }
sep()  { echo -e "${BLUE}────────────────────────────────────────${NC}"; }

# ── IST time helpers ──────────────────────────────────────────────────────────
ist_time() {
    TZ="Asia/Kolkata" date '+%H:%M'
}

ist_display() {
    TZ="Asia/Kolkata" date '+%H:%M:%S IST (%a %d %b %Y)'
}

market_status() {
    local t
    t=$(ist_time)
    local dow
    dow=$(TZ="Asia/Kolkata" date '+%u')  # 1=Mon … 7=Sun

    if [ "$dow" -ge 6 ]; then
        echo "closed"
    elif [[ "$t" > "09:00" && "$t" < "09:15" ]]; then
        echo "pre_open"
    elif [[ "$t" > "09:14" && "$t" < "15:31" ]]; then
        echo "open"
    elif [[ "$t" > "15:30" && "$t" < "18:01" ]]; then
        echo "post_market"
    else
        echo "closed"
    fi
}

next_open() {
    local dow
    dow=$(TZ="Asia/Kolkata" date '+%u')
    if   [ "$dow" -eq 5 ]; then TZ="Asia/Kolkata" date -v+3d '+%a %d %b 09:15 IST' 2>/dev/null || TZ="Asia/Kolkata" date -d '+3 days' '+%a %d %b 09:15 IST'
    elif [ "$dow" -eq 6 ]; then TZ="Asia/Kolkata" date -v+2d '+%a %d %b 09:15 IST' 2>/dev/null || TZ="Asia/Kolkata" date -d '+2 days' '+%a %d %b 09:15 IST'
    else                        TZ="Asia/Kolkata" date -v+1d '+%a %d %b 09:15 IST' 2>/dev/null || TZ="Asia/Kolkata" date -d '+1 day'  '+%a %d %b 09:15 IST'
    fi
}

# ── DB stats ──────────────────────────────────────────────────────────────────
db_stats() {
    if [ ! -f "$PROJECT_DIR/data.db" ]; then
        warn "data.db not found"
        return
    fi
    local snap candles opt_candles
    snap=$(sqlite3 "$PROJECT_DIR/data.db" "SELECT COUNT(*) FROM option_chain_snapshot" 2>/dev/null || echo 0)
    candles=$(sqlite3 "$PROJECT_DIR/data.db" "SELECT COUNT(*) FROM historical_candle" 2>/dev/null || echo 0)
    opt_candles=$(sqlite3 "$PROJECT_DIR/data.db" "SELECT COUNT(*) FROM historical_option" 2>/dev/null || echo 0)
    echo "  DB: snapshots=$snap  index_candles=$candles  option_candles=$opt_candles"
}

model_status() {
    local found=0
    for asset in nifty banknifty finnifty midcpnifty; do
        if [ -f "$PROJECT_DIR/trained_models/${asset}_ensemble.pkl" ]; then
            ok "Model: $asset"
            found=$((found+1))
        else
            warn "No model: $asset"
        fi
    done
    echo "  $found/4 models trained"
}

# ── Actions ───────────────────────────────────────────────────────────────────

do_status() {
    sep
    echo -e "${BOLD}Options Intelligence Platform — Status${NC}"
    sep
    echo "  IST Time:      $(ist_display)"
    echo "  Market:        $(market_status | tr '[:lower:]' '[:upper:]')"
    echo "  Next open:     $(next_open 2>/dev/null || echo 'N/A')"
    sep
    db_stats
    sep
    model_status
    sep
}

do_collect() {
    local mkt_status
    mkt_status=$(market_status)
    log "Market mkt_status: ${BOLD}${mkt_status}${NC}"

    case "$mkt_status" in
        open|pre_open)
            log "Market is OPEN — starting live collection (every 60s)"
            log "Will auto-switch to EOD at 15:30"
            cd "$PROJECT_DIR"
            "$PYTHON_DATA" collect_data.py --live 60
            ;;
        post_market)
            log "Post-market — collecting EOD data"
            cd "$PROJECT_DIR"
            # Run directly (no tee) so OTP prompt is visible
            "$PYTHON_DATA" collect_data.py --eod
            ok "EOD collection done"
            ;;
        *)
            warn "Market closed. Fetching historical data only."
            warn "Next open: $(next_open 2>/dev/null || echo 'N/A')"
            cd "$PROJECT_DIR"
            "$PYTHON_DATA" collect_data.py --historical
            ok "Historical collection done"
            ;;
    esac
}

do_train() {
    log "Training ML models (CatBoost + XGBoost + LightGBM + LSTM + Transformer)..."
    cd "$PROJECT_DIR"
    OMP_NUM_THREADS=1 "$PYTHON_TRAIN" build_model.py NIFTY BANKNIFTY FINNIFTY MIDCPNIFTY \
        2>&1 | tee -a "$LOG_DIR/training.log"
    ok "Training complete"
    model_status
}

do_server() {
    log "Starting FastAPI server on http://localhost:8000"
    log "Open frontend: file://$PROJECT_DIR/frontend/index.html"
    cd "$PROJECT_DIR"
    OMP_NUM_THREADS=1 "$UVICORN" vol_server:app --host 0.0.0.0 --port 8000 \
        2>&1 | tee -a "$LOG_DIR/server.log"
}

do_all() {
    sep
    log "Running full pipeline: collect → train → server"
    sep
    do_collect
    sep
    do_train
    sep
    do_server
}

do_schedule() {
    sep
    log "Scheduler mode — runs continuously, adapts to market hours"
    log "Press Ctrl+C to stop"
    sep

    while true; do
        local mkt_status
        mkt_status=$(market_status)
        local t
        t=$(ist_time)

        case "$mkt_status" in
            pre_open)
                log "Pre-open ($t IST) — fetching historical, waiting for open..."
                cd "$PROJECT_DIR"
                "$PYTHON_DATA" collect_data.py --historical >> "$LOG_DIR/collector.log" 2>&1
                sleep 300  # check again in 5 min
                ;;

            open)
                log "Market OPEN ($t IST) — live collection"
                cd "$PROJECT_DIR"
                "$PYTHON_DATA" collect_data.py --live 60 >> "$LOG_DIR/collector.log" 2>&1
                # --live 60 exits automatically when market closes
                ;;

            post_market)
                log "Post-market ($t IST) — EOD collection + model retrain"
                cd "$PROJECT_DIR"
                "$PYTHON_DATA" collect_data.py --eod >> "$LOG_DIR/collector.log" 2>&1
                ok "EOD done"
                db_stats

                # Retrain models with today's new data
                log "Retraining models with today's data..."
                OMP_NUM_THREADS=1 "$PYTHON_TRAIN" build_model.py NIFTY BANKNIFTY FINNIFTY MIDCPNIFTY \
                    >> "$LOG_DIR/training.log" 2>&1
                ok "Models retrained"

                log "Waiting for market close window to pass..."
                sleep 3600  # wait 1 hour before checking again
                ;;

            closed)
                local next
                next=$(next_open 2>/dev/null || echo "tomorrow")
                log "Market CLOSED ($t IST) — next open: $next"
                log "Sleeping 30 minutes..."
                sleep 1800
                ;;
        esac
    done
}

do_auto() {
    # Default: smart single run based on current time
    local mkt_status
    mkt_status=$(market_status)
    sep
    echo -e "${BOLD}Options Intelligence Platform${NC}"
    echo "  Time:   $(ist_display)"
    echo "  Market: ${BOLD}${mkt_status}${NC}"
    sep

    case "$mkt_status" in
        open|pre_open)
            log "Market is OPEN — what would you like to do?"
            echo "  1) Start live data collection (--live 60)"
            echo "  2) Start server"
            echo "  3) Both (collect in background + server)"
            echo "  4) Just show mkt_status"
            print -n "Choice [1-4]: "; read -r choice
            case "$choice" in
                1) do_collect ;;
                2) do_server ;;
                3)
                    log "Starting collector in background..."
                    cd "$PROJECT_DIR"
                    "$PYTHON_DATA" collect_data.py --live 60 >> "$LOG_DIR/collector.log" 2>&1 &
                    ok "Collector running (PID $!)"
                    do_server
                    ;;
                *) do_status ;;
            esac
            ;;

        post_market)
            log "Post-market window — recommended: collect EOD + retrain"
            echo "  1) Collect EOD data"
            echo "  2) Collect EOD + retrain models"
            echo "  3) Start server (use existing data)"
            echo "  4) Full pipeline (EOD + train + server)"
            print -n "Choice [1-4]: "; read -r choice
            case "$choice" in
                1) do_collect ;;
                2) do_collect; do_train ;;
                3) do_server ;;
                4) do_all ;;
                *) do_status ;;
            esac
            ;;

        closed)
            warn "Market is CLOSED. Next open: $(next_open 2>/dev/null || echo 'N/A')"
            echo "  1) Fetch historical data"
            echo "  2) Train models (use existing data)"
            echo "  3) Start server (use existing data)"
            echo "  4) Show mkt_status"
            print -n "Choice [1-4]: "; read -r choice
            case "$choice" in
                1) do_collect ;;
                2) do_train ;;
                3) do_server ;;
                *) do_status ;;
            esac
            ;;
    esac
}

# ── Entry point ───────────────────────────────────────────────────────────────
CMD="${1:-auto}"

case "$CMD" in
    server)   do_server ;;
    collect)  do_collect ;;
    train)    do_train ;;
    all)      do_all ;;
    schedule) do_schedule ;;
    status)   do_status ;;
    auto|"")  do_auto ;;
    *)
        echo "Usage: ./run.sh [server|collect|train|all|schedule|status]"
        echo "       ./run.sh          → interactive auto-detect"
        exit 1
        ;;
esac
