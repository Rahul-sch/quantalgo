#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_dashboard.sh — Boot Streamlit + Cloudflare Tunnel
#
# Usage:  ./launch_dashboard.sh
# Stops:  Ctrl+C  (kills both Streamlit and the tunnel cleanly)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8501
STREAMLIT_LOG="/tmp/streamlit_dashboard.log"
TUNNEL_LOG="/tmp/cloudflare_tunnel.log"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   🏛️  QQQ ALGO DASHBOARD — LAUNCH SEQUENCE       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${YELLOW}  Shutting down...${RESET}"
    [[ -n "${STREAMLIT_PID:-}" ]] && kill "$STREAMLIT_PID" 2>/dev/null && echo "  ✅ Streamlit stopped"
    [[ -n "${TUNNEL_PID:-}"    ]] && kill "$TUNNEL_PID"    2>/dev/null && echo "  ✅ Tunnel closed"
    echo -e "${GREEN}  Goodbye. Run again tomorrow at 9:30 AM ET 🎯${RESET}"
    echo ""
    exit 0
}
trap cleanup INT TERM

# ── Kill any existing Streamlit on this port ─────────────────────────────────
EXISTING=$(lsof -ti tcp:$PORT 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    echo -e "${YELLOW}  ⚠️  Port $PORT already in use — killing existing process...${RESET}"
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
fi

# ── Step 1: Start Streamlit ───────────────────────────────────────────────────
echo -e "  ${CYAN}[1/3]${RESET} Starting Streamlit on port $PORT..."
cd "$SCRIPT_DIR"
python3 -m streamlit run dashboard.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    > "$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!

# Wait for Streamlit to be ready
echo -n "        Waiting for Streamlit"
for i in $(seq 1 20); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/" 2>/dev/null | grep -q "200"; then
        echo -e " ${GREEN}✅${RESET}"
        break
    fi
    echo -n "."
    sleep 1
    if [[ $i -eq 20 ]]; then
        echo -e " ${RED}❌ Streamlit failed to start${RESET}"
        echo "  Check log: $STREAMLIT_LOG"
        cat "$STREAMLIT_LOG" | tail -20
        exit 1
    fi
done

# ── Step 2: Start Cloudflare Tunnel ──────────────────────────────────────────
echo -e "  ${CYAN}[2/3]${RESET} Opening Cloudflare Tunnel..."
cloudflared tunnel --url "http://localhost:$PORT" \
    --no-autoupdate \
    > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Extract the public URL (Cloudflare prints it within ~5 seconds)
echo -n "        Waiting for tunnel URL"
PUBLIC_URL=""
for i in $(seq 1 30); do
    PUBLIC_URL=$(grep -oE 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
    if [[ -n "$PUBLIC_URL" ]]; then
        echo -e " ${GREEN}✅${RESET}"
        break
    fi
    echo -n "."
    sleep 1
    if [[ $i -eq 30 ]]; then
        echo -e " ${RED}❌ Could not get tunnel URL${RESET}"
        echo "  Check log: $TUNNEL_LOG"
        cat "$TUNNEL_LOG" | tail -20
        cleanup
    fi
done

# ── Step 3: Print the dashboard URL ──────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   ${GREEN}[3/3] DASHBOARD IS LIVE${RESET}${BOLD}                         ║${RESET}"
echo -e "${BOLD}╠══════════════════════════════════════════════════╣${RESET}"
echo -e "${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  📱 ${BOLD}Phone / Remote URL:${RESET}"
echo -e "${BOLD}║${RESET}     ${CYAN}${BOLD}${PUBLIC_URL}${RESET}"
echo -e "${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  💻 Local URL:"
echo -e "${BOLD}║${RESET}     http://localhost:$PORT"
echo -e "${BOLD}║${RESET}"
echo -e "${BOLD}║${RESET}  ℹ️  Press ${BOLD}Ctrl+C${RESET} to stop both servers"
echo -e "${BOLD}║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Keep alive + auto-print URL if it rotates ────────────────────────────────
echo -e "  ${YELLOW}Monitoring tunnel... (Ctrl+C to stop)${RESET}"
while true; do
    # Check Streamlit is still alive
    if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        # Kill anything still on the port before restarting
        STALE=$(lsof -ti tcp:$PORT 2>/dev/null || true)
        [[ -n "$STALE" ]] && kill "$STALE" 2>/dev/null && sleep 1
        echo -e "${RED}  ⚠️  Streamlit crashed — restarting...${RESET}"
        python3 -m streamlit run dashboard.py \
            --server.port "$PORT" \
            --server.headless true \
            > "$STREAMLIT_LOG" 2>&1 &
        STREAMLIT_PID=$!
        sleep 3
    fi
    # Check tunnel is still alive
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo -e "${RED}  ⚠️  Tunnel dropped — check $TUNNEL_LOG${RESET}"
        break
    fi
    sleep 10
done

cleanup
