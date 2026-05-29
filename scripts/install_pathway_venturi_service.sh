#!/usr/bin/env bash
# Install Pathway Venturi intake as a systemd user service.
# Watches KRAMPUSCHEWING 24/7, streams files to Needle swarm, stages GO-25 candidates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PYTHON="${VENV}/bin/python3"
SERVICE_NAME="lucidota-venturi-intake"

echo "[install] Checking venv at ${VENV}..."
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: venv not found at ${VENV}. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "[install] Installing Pathway + deps into venv..."
"${PYTHON}" -m pip install --quiet pathway python-magic pdfminer.six psycopg httpx

echo "[install] Checking deps..."
"${PYTHON}" "${SCRIPT_DIR}/pathway_venturi_intake.py" --check-deps

# Create systemd user service
SERVICE_FILE="${HOME}/.config/systemd/user/${SERVICE_NAME}.service"
mkdir -p "$(dirname "${SERVICE_FILE}")"

cat > "${SERVICE_FILE}" <<SERVICE
[Unit]
Description=Lucidota Pathway Venturi Intake — KRAMPUSCHEWING corpus pipe to GO-25 staging
After=network.target postgresql.service

[Service]
Type=simple
WorkingDirectory=${REPO_ROOT}
Environment="PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin"
Environment="VENTURI_SOURCE_DIR=${REPO_ROOT}/KRAMPUSCHEWING"
EnvironmentFile=-${HOME}/.config/lucidota/secrets.env
ExecStart=${PYTHON} ${SCRIPT_DIR}/pathway_venturi_intake.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal+console
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=default.target
SERVICE

echo "[install] Service file written to ${SERVICE_FILE}"

systemctl --user daemon-reload
systemctl --user enable "${SERVICE_NAME}.service"
echo "[install] Enabled ${SERVICE_NAME}"
echo ""
echo "To start: systemctl --user start ${SERVICE_NAME}"
echo "To watch: journalctl --user -u ${SERVICE_NAME} -f"
echo ""
echo "NOTE: Needles must be running on :8090-:8095 for GO-25 extraction."
echo "      Files will still be dedup-recorded even if Needles are offline."
