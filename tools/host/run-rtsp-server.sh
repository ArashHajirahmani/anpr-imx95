#!/bin/bash

# Publish local MP4 files so the i.MX95 board can read them over RTSP.
# The board URL is rtsp://<this-pc>:8554/videoCapture

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STREAM_PATH="videoCapture"
RTSP_PORT=8554
PUBLISH_HOST=127.0.0.1
SEEK_STEP_SEC=10

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
    echo "Keyboard (foreground terminal only):"
    echo "  Space     pause / resume (SIGSTOP / SIGCONT on ffmpeg)"
    echo "  → / ←     seek +${SEEK_STEP_SEC}s / -${SEEK_STEP_SEC}s (restarts ffmpeg at -ss)"
    echo "  q         quit"
    echo "  Seeking reconnects the RTSP publisher; the board may drop a frame"
    echo "  or two, then the capture thread opens the stream again."
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

INTERACTIVE=0
if [[ -t 0 ]]; then
    INTERACTIVE=1
fi

echo "============================================================"
echo "Found ${#mp4_files[@]} .mp4 file(s)."
for f in "${mp4_files[@]}"; do
    echo "  - $f"
done
echo "Listen: 0.0.0.0:${RTSP_PORT}"
echo "Board URL: rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"
if [[ "$INTERACTIVE" -eq 1 ]]; then
    echo "Keys: Space=pause  ←/→=seek ±${SEEK_STEP_SEC}s  q=quit"
else
    echo "Keys: disabled (stdin is not a terminal; run in the foreground)"
fi
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
        kill -CONT "$FFMPEG_PID" 2>/dev/null || true
        kill "$FFMPEG_PID" 2>/dev/null || true
        wait "$FFMPEG_PID" 2>/dev/null || true
    fi
    if [[ -n "$MEDIAMTX_PID" ]]; then
        kill "$MEDIAMTX_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

file_duration() {
    local file=$1
    local seconds
    seconds=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$file" 2>/dev/null || true)
    if [[ -z "$seconds" ]]; then
        echo 0
        return
    fi
    printf '%.0f' "$seconds"
}

start_ffmpeg() {
    local file=$1
    local start_sec=$2
    local args=(-hide_banner -loglevel warning -re)
    if [[ "$start_sec" -gt 0 ]]; then
        args+=(-ss "$start_sec")
    fi
    args+=(-i "$file" -c:v copy -an -rtsp_transport tcp -f rtsp
        "rtsp://${PUBLISH_HOST}:${RTSP_PORT}/${STREAM_PATH}")
    ffmpeg "${args[@]}" &
    FFMPEG_PID=$!
    PLAY_ORIGIN=$start_sec
    PLAY_STARTED_AT=$(date +%s)
    PAUSED=0
}

current_position() {
    local now elapsed
    if [[ "$PAUSED" -eq 1 ]]; then
        echo "$FROZEN_POS"
        return
    fi
    now=$(date +%s)
    elapsed=$((now - PLAY_STARTED_AT))
    echo $((PLAY_ORIGIN + elapsed))
}

toggle_pause() {
    if [[ -z "$FFMPEG_PID" ]] || ! kill -0 "$FFMPEG_PID" 2>/dev/null; then
        return
    fi
    if [[ "$PAUSED" -eq 0 ]]; then
        FROZEN_POS=$(current_position)
        kill -STOP "$FFMPEG_PID" 2>/dev/null || true
        PAUSED=1
        echo "[PAUSED] at ${FROZEN_POS}s"
    else
        kill -CONT "$FFMPEG_PID" 2>/dev/null || true
        PLAY_ORIGIN=$FROZEN_POS
        PLAY_STARTED_AT=$(date +%s)
        PAUSED=0
        echo "[RESUME] from ${PLAY_ORIGIN}s"
    fi
}

seek_by() {
    local file=$1
    local delta=$2
    local duration=$3
    local pos new_pos
    pos=$(current_position)
    new_pos=$((pos + delta))
    if [[ "$new_pos" -lt 0 ]]; then
        new_pos=0
    fi
    if [[ "$duration" -gt 0 && "$new_pos" -ge "$duration" ]]; then
        echo "[SEEK] past end (${duration}s); advancing to next file"
        if [[ -n "$FFMPEG_PID" ]]; then
            kill -CONT "$FFMPEG_PID" 2>/dev/null || true
            kill "$FFMPEG_PID" 2>/dev/null || true
            wait "$FFMPEG_PID" 2>/dev/null || true
            FFMPEG_PID=""
        fi
        return 1
    fi
    echo "[SEEK] ${pos}s -> ${new_pos}s"
    if [[ -n "$FFMPEG_PID" ]]; then
        kill -CONT "$FFMPEG_PID" 2>/dev/null || true
        kill "$FFMPEG_PID" 2>/dev/null || true
        wait "$FFMPEG_PID" 2>/dev/null || true
        FFMPEG_PID=""
    fi
    # Brief gap so MediaMTX drops the old publisher before the new one.
    sleep 0.3
    start_ffmpeg "$file" "$new_pos"
    return 0
}

read_key() {
    # Returns in REPLY: space|left|right|quit|other
    local key rest
    if ! read -rsn1 -t 0.2 key; then
        REPLY=""
        return 1
    fi
    case "$key" in
        " ")
            REPLY=space
            ;;
        q|Q)
            REPLY=quit
            ;;
        $'\x1b')
            # Arrow keys: ESC [ A/B/C/D
            read -rsn1 -t 0.05 rest || true
            if [[ "$rest" == "[" ]]; then
                read -rsn1 -t 0.05 rest || true
                case "$rest" in
                    C) REPLY=right ;;
                    D) REPLY=left ;;
                    *) REPLY=other ;;
                esac
            else
                REPLY=other
            fi
            ;;
        *)
            REPLY=other
            ;;
    esac
    return 0
}

publish_file_once() {
    local file=$1
    local duration
    duration=$(file_duration "$file")
    echo "[STREAMING] $file -> rtsp://${BOARD_HOST}:${RTSP_PORT}/${STREAM_PATH}"
    if [[ "$duration" -gt 0 ]]; then
        echo "[INFO] duration ${duration}s"
    fi
    start_ffmpeg "$file" 0

    if [[ "$INTERACTIVE" -eq 0 ]]; then
        wait "$FFMPEG_PID" || echo "Publish stopped. Retrying ${STREAM_PATH}."
        FFMPEG_PID=""
        return 0
    fi

    while [[ -n "$FFMPEG_PID" ]] && kill -0 "$FFMPEG_PID" 2>/dev/null; do
        if read_key; then
            case "$REPLY" in
                space)
                    toggle_pause
                    ;;
                right)
                    seek_by "$file" "$SEEK_STEP_SEC" "$duration" || return 0
                    ;;
                left)
                    seek_by "$file" "-${SEEK_STEP_SEC}" "$duration" || true
                    ;;
                quit)
                    cleanup
                    ;;
            esac
        fi
        if [[ -n "$FFMPEG_PID" ]] && ! kill -0 "$FFMPEG_PID" 2>/dev/null; then
            wait "$FFMPEG_PID" 2>/dev/null || true
            FFMPEG_PID=""
            break
        fi
    done
}

while true; do
    for file in "${mp4_files[@]}"; do
        publish_file_once "$file"
        sleep 1
    done
done
