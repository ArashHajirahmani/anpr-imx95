#!/bin/bash

# Publish local MP4 files so the i.MX95 board can read them over RTSP.
# The board URL is rtsp://<this-pc>:8554/videoCapture

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STREAM_PATH="videoCapture"
RTSP_PORT=8554
PUBLISH_HOST=127.0.0.1

show_help() {
    echo "============================================================"
    echo " RTSP MP4 publisher for the i.MX95 ANPR bench"
    echo "============================================================"
    echo "Usage: $0 [DIRECTORY|FILE.mp4]"
    echo ""
    echo "Description:"
    echo "  Accepts a directory of .mp4 files or a single .mp4 file."
    echo "  Starts MediaMTX on 0.0.0.0:${RTSP_PORT} and publishes H.264"
    echo "  video forever on path ${STREAM_PATH}."
    echo "  The board opens rtsp://<this-pc>:${RTSP_PORT}/${STREAM_PATH}"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - ffmpeg on PATH"
    echo "  - MediaMTX executable in this directory, next to this script:"
    echo "      ${SCRIPT_DIR}/mediamtx"
    echo "    Other files from the release can sit here too (mediamtx.yml, LICENSE,"
    echo "    auto.crt, auto.key). They are gitignored. MEDIAMTX_BIN or a mediamtx"
    echo "    binary on PATH also works."
    echo "  - Video tracks must be H.264. The board decodes with v4l2h264dec."
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help    Show this help"
    echo ""
    echo "EXAMPLE:"
    echo "  $0 datasets/videos"
    echo "  $0 datasets/videos/IMG_9475.mp4"
    echo "============================================================"
    exit 0
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
fi

TARGET="${1:-.}"

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

collect_mp4_files() {
    local target=$1
    mp4_files=()
    if [[ -f "$target" ]]; then
        case "$target" in
            *.mp4|*.MP4) ;;
            *)
                echo "Error: file must be a .mp4: $target" >&2
                return 1
                ;;
        esac
        mp4_files+=("$target")
        return 0
    fi
    if [[ -d "$target" ]]; then
        local f
        shopt -s nullglob
        for f in "$target"/*.mp4 "$target"/*.MP4; do
            [[ -e "$f" ]] || continue
            mp4_files+=("$f")
        done
        shopt -u nullglob
        if [[ ${#mp4_files[@]} -eq 0 ]]; then
            echo "Error: No .mp4 files found in '$target'." >&2
            return 1
        fi
        return 0
    fi
    echo "Error: not a directory or .mp4 file: $target" >&2
    return 1
}

if ! collect_mp4_files "$TARGET"; then
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
for f in "${mp4_files[@]}"; do
    echo "  - $f"
done
echo "Listen: 0.0.0.0:${RTSP_PORT}"
echo "Board URL: rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"
echo "============================================================"

# True when :8554 is open on every interface. A 127.0.0.1 listener is not enough.
rtsp_listens_everywhere() {
    ss -ltn "sport = :${RTSP_PORT}" | awk 'NR > 1 { print $4 }' | grep -Eq \
        '(^|[^0-9])(\*|0\.0\.0\.0|\[::\]):'"${RTSP_PORT}"'$'
}

wait_for_rtsp() {
    local i
    for i in $(seq 1 20); do
        if rtsp_listens_everywhere; then
            return 0
        fi
        sleep 0.25
    done
    return 1
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
    # Keep MediaMTX logs out of the foreground so Ctrl+C hits this script cleanly.
    "$MEDIAMTX" "$config" >/tmp/anpr-mediamtx.log 2>&1 &
    MEDIAMTX_PID=$!
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

if ! wait_for_rtsp; then
    echo "Error: nothing is listening on 0.0.0.0:${RTSP_PORT}." >&2
    echo "MediaMTX log: /tmp/anpr-mediamtx.log" >&2
    if [[ -n "$MEDIAMTX_PID" ]]; then
        kill "$MEDIAMTX_PID" 2>/dev/null || true
    fi
    exit 1
fi

if pgrep -af ffmpeg | grep -q "${RTSP_PORT}/${STREAM_PATH}"; then
    echo "Error: ${STREAM_PATH} already has a publisher. Stop that ffmpeg before starting another." >&2
    exit 1
fi

FFMPEG_PID=""
cleanup() {
    echo ""
    echo "Shutting down the ${STREAM_PATH} publisher..."
    if [[ -n "$FFMPEG_PID" ]]; then
        kill "$FFMPEG_PID" 2>/dev/null || true
    fi
    if [[ -n "$MEDIAMTX_PID" ]]; then
        kill "$MEDIAMTX_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    for file in "${mp4_files[@]}"; do
        echo "[STREAMING] $file -> rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"
        # Publish to loopback; the board still reads via the LAN address above.
        ffmpeg -hide_banner -loglevel warning -re -i "$file" -c:v copy -an \
            -rtsp_transport tcp -f rtsp \
            "rtsp://${PUBLISH_HOST}:${RTSP_PORT}/${STREAM_PATH}" &
        FFMPEG_PID=$!
        wait "$FFMPEG_PID" || echo "Publish stopped. Retrying ${STREAM_PATH}."
        FFMPEG_PID=""
        sleep 1
    done
done
