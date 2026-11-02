#!/bin/bash

# Publish local MP4 files so the i.MX95 board can read them over RTSP.
# The board URL is rtsp://<this-pc>:8554/videoCapture

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STREAM_PATH="videoCapture"
RTSP_PORT=8554

show_help() {
    echo "============================================================"
    echo " RTSP multi-file MP4 publisher for the i.MX95 ANPR bench"
    echo "============================================================"
    echo "Usage: $0 [DIRECTORY]"
    echo ""
    echo "Description:"
    echo "  Scans DIRECTORY for .mp4 files, starts MediaMTX on 0.0.0.0:${RTSP_PORT},"
    echo "  and publishes each file in order, forever, on the ${STREAM_PATH} path."
    echo "  The board opens rtsp://<this-pc>:${RTSP_PORT}/${STREAM_PATH}"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - ffmpeg on PATH"
    echo "  - MediaMTX executable in this directory, next to this script:"
    echo "      ${SCRIPT_DIR}/mediamtx"
    echo "    Other files from the release can sit here too (mediamtx.yml, LICENSE,"
    echo "    auto.crt, auto.key). They are gitignored. MEDIAMTX_BIN or a mediamtx"
    echo "    binary on PATH also works."
    echo "  - At least one .mp4 file in the target directory"
    echo "  - Video tracks must be H.264. The board decodes with v4l2h264dec."
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help    Show this help"
    echo ""
    echo "EXAMPLE:"
    echo "  $0 /path/to/videos"
    echo "============================================================"
    exit 0
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
fi

TARGET_DIR="${1:-.}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: 'ffmpeg' is required but not found in PATH." >&2
    exit 1
fi

if [[ -n "${MEDIAMTX_BIN:-}" ]]; then
    MEDIAMTX="$MEDIAMTX_BIN"
elif [[ -x "$SCRIPT_DIR/mediamtx" ]]; then
    MEDIAMTX="$SCRIPT_DIR/mediamtx"
elif command -v mediamtx >/dev/null 2>&1; then
    MEDIAMTX=$(command -v mediamtx)
else
    echo "Error: mediamtx binary not found." >&2
    echo "Put the executable at ${SCRIPT_DIR}/mediamtx (this directory), or set MEDIAMTX_BIN." >&2
    exit 1
fi

if [[ ! -x "$MEDIAMTX" ]]; then
    echo "Error: MediaMTX is not executable: $MEDIAMTX" >&2
    exit 1
fi

shopt -s nullglob
mp4_files=("$TARGET_DIR"/*.mp4)
shopt -u nullglob

if [[ ${#mp4_files[@]} -eq 0 ]]; then
    echo "Error: No .mp4 files found in '$TARGET_DIR'." >&2
    exit 1
fi

lan_ip() {
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{
        for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }
    }'
}

BOARD_HOST=$(lan_ip)
if [[ -z "$BOARD_HOST" ]]; then
    BOARD_HOST="<this-pc>"
fi

echo "============================================================"
echo "Found ${#mp4_files[@]} .mp4 file(s)."
echo "Listen: 0.0.0.0:${RTSP_PORT}"
echo "Board URL: rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"
echo "============================================================"

# True when :8554 is open on every interface. A 127.0.0.1 listener is not enough.
rtsp_listens_everywhere() {
    ss -ltn "sport = :${RTSP_PORT}" | awk 'NR > 1 { print $4 }' | grep -Eq \
        '(^|[^0-9])(\*|0\.0\.0\.0|\[::\]):'"${RTSP_PORT}"'$'
}

start_mediamtx() {
    local config
    config=$(mktemp)
    cat >"$config" <<EOF
rtsp: true
rtspAddress: 0.0.0.0:${RTSP_PORT}
rtspTransports: [tcp]
authInternalUsers:
  - user: any
    pass:
    ips: []
    permissions:
      - action: publish
      - action: read
      - action: playback
paths:
  all_others:
EOF
    echo "Starting MediaMTX from $MEDIAMTX (listening on 0.0.0.0:${RTSP_PORT})..."
    "$MEDIAMTX" "$config" &
    MEDIAMTX_PID=$!
    sleep 1
}

MEDIAMTX_PID=""
if pgrep -x "mediamtx" >/dev/null && rtsp_listens_everywhere; then
    echo "MediaMTX is already listening on 0.0.0.0:${RTSP_PORT}."
elif pgrep -x "mediamtx" >/dev/null; then
    echo "MediaMTX is not listening on 0.0.0.0:${RTSP_PORT}. Restarting it."
    pkill -x mediamtx || true
    sleep 0.5
    start_mediamtx
else
    start_mediamtx
fi

if ! rtsp_listens_everywhere; then
    echo "Error: nothing is listening on 0.0.0.0:${RTSP_PORT}." >&2
    exit 1
fi

if pgrep -af ffmpeg | grep -q "${RTSP_PORT}/${STREAM_PATH}"; then
    echo "Error: ${STREAM_PATH} already has a publisher. Stop that ffmpeg before starting another." >&2
    exit 1
fi

cleanup() {
    echo ""
    echo "Shutting down the ${STREAM_PATH} publisher..."
    if [[ -n "$MEDIAMTX_PID" ]]; then
        kill "$MEDIAMTX_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    for file in "${mp4_files[@]}"; do
        echo "[STREAMING] $file -> rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"
        if ! ffmpeg -hide_banner -loglevel warning -re -i "$file" -c:v copy -an \
            -rtsp_transport tcp -f rtsp "rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"; then
            echo "Publish stopped. Retrying ${STREAM_PATH}."
        fi
        sleep 1
    done
done
