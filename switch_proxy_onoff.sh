#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  switch_proxy_onoff.sh
#  Toggle between Ollama and ollama-proxy.
#  - If ollama-proxy service doesn't exist yet, creates and starts it.
#  - If ollama is active  → stops/disables it, starts ollama-proxy.
#  - If ollama-proxy is active → stops it, re-enables ollama.
#
#  Usage:  sudo ./switch_proxy_onoff.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── config — edit these to match your setup ───────────────────────────────────
PROXY_USER="YOUR_USERNAME"
PROXY_DIR="/path/to/ollama-proxy"
PROXY_SCRIPT="$PROXY_DIR/ollama_proxy.py"
PYTHON_BIN="/usr/bin/python"          # or /path/to/venv/bin/python
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_OLLAMA="ollama"
SERVICE_PROXY="ollama-proxy"
UNIT_FILE="/etc/systemd/system/${SERVICE_PROXY}.service"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
title() { echo -e "\n${BOLD}${CYAN}$*${NC}"; }

[[ $EUID -eq 0 ]] || { echo "Run as root: sudo ./switch_proxy_onoff.sh"; exit 1; }

# ── helpers ───────────────────────────────────────────────────────────────────

service_exists() {
    systemctl cat "$1" &>/dev/null 2>&1
}

service_active() {
    systemctl is-active --quiet "$1" 2>/dev/null
}

service_enabled() {
    systemctl is-enabled --quiet "$1" 2>/dev/null
}

# ── create the proxy service unit if missing ──────────────────────────────────

create_proxy_service() {
    title "Creating ollama-proxy systemd service…"

    [[ -f "$PROXY_SCRIPT" ]] || {
        echo "  ✗ Proxy script not found at $PROXY_SCRIPT"
        echo "    Edit PROXY_DIR at the top of this script."
        exit 1
    }

    cat > "$UNIT_FILE" << UNIT
[Unit]
Description=Ollama → llama.cpp Proxy
After=network.target

[Service]
Type=simple
User=${PROXY_USER}
WorkingDirectory=${PROXY_DIR}
ExecStart=${PYTHON_BIN} ${PROXY_SCRIPT}
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
UNIT

    systemctl daemon-reload
    info "Service unit created at $UNIT_FILE"
}

# ── print current state ───────────────────────────────────────────────────────

print_state() {
    title "Current state"
    local ollama_state proxy_state
    service_active "$SERVICE_OLLAMA" && ollama_state="${GREEN}RUNNING${NC}" || ollama_state="stopped"
    service_active "$SERVICE_PROXY"  && proxy_state="${GREEN}RUNNING${NC}"  || proxy_state="stopped"
    echo -e "  ollama        : $ollama_state"
    echo -e "  ollama-proxy  : $proxy_state"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════

print_state

# Create service unit if it doesn't exist yet
service_exists "$SERVICE_PROXY" || create_proxy_service

# ── decide which way to toggle ────────────────────────────────────────────────

if service_active "$SERVICE_PROXY"; then
    # ── proxy is running → switch back to Ollama ─────────────────────────────
    title "Switching to Ollama (stopping proxy)…"

    info "Stopping ollama-proxy…"
    systemctl stop "$SERVICE_PROXY"
    systemctl disable "$SERVICE_PROXY" 2>/dev/null || true

    if service_exists "$SERVICE_OLLAMA"; then
        info "Starting ollama…"
        systemctl enable "$SERVICE_OLLAMA" 2>/dev/null || true
        systemctl start "$SERVICE_OLLAMA"
        info "ollama is now active ✓"
    else
        warn "ollama service not found — install Ollama to use it."
        warn "  curl -fsSL https://ollama.com/install.sh | sh"
    fi

    echo ""
    echo -e "  ${BOLD}Active now:${NC}  Ollama  (port 11434)"
    echo -e "  ${BOLD}Stopped:${NC}     ollama-proxy"

elif service_active "$SERVICE_OLLAMA"; then
    # ── ollama is running → switch to proxy ──────────────────────────────────
    title "Switching to ollama-proxy (stopping Ollama)…"

    info "Stopping ollama…"
    systemctl stop "$SERVICE_OLLAMA"
    systemctl disable "$SERVICE_OLLAMA" 2>/dev/null || true

    info "Starting ollama-proxy…"
    systemctl enable "$SERVICE_PROXY"
    systemctl start "$SERVICE_PROXY"

    info "Waiting for proxy to come up (model load can take ~60s)…"
    DEADLINE=$((SECONDS + 180))
    until curl -sf http://127.0.0.1:11434/health >/dev/null 2>&1; do
        [[ $SECONDS -lt $DEADLINE ]] || {
            warn "Proxy did not respond within 180s."
            warn "Check logs:  journalctl -u ollama-proxy -f"
            exit 1
        }
        sleep 2
    done

    info "ollama-proxy is now active ✓"
    echo ""
    echo -e "  ${BOLD}Active now:${NC}  ollama-proxy  (port 11434, llama.cpp backend)"
    echo -e "  ${BOLD}Stopped:${NC}     Ollama"

else
    # ── neither is running → start proxy as default ──────────────────────────
    title "Neither service is running — starting ollama-proxy…"

    # Make sure ollama is stopped/disabled so nothing fights for port 11434
    if service_exists "$SERVICE_OLLAMA"; then
        systemctl stop  "$SERVICE_OLLAMA" 2>/dev/null || true
        systemctl disable "$SERVICE_OLLAMA" 2>/dev/null || true
    fi

    systemctl enable "$SERVICE_PROXY"
    systemctl start  "$SERVICE_PROXY"

    info "Waiting for proxy to come up…"
    DEADLINE=$((SECONDS + 180))
    until curl -sf http://127.0.0.1:11434/health >/dev/null 2>&1; do
        [[ $SECONDS -lt $DEADLINE ]] || {
            warn "Proxy did not respond. Check:  journalctl -u ollama-proxy -f"
            exit 1
        }
        sleep 2
    done

    info "ollama-proxy started ✓"
    echo ""
    echo -e "  ${BOLD}Active now:${NC}  ollama-proxy  (port 11434)"
fi

echo ""
print_state
