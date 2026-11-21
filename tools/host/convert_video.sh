#!/bin/bash

# Convert a phone video (HEVC .MOV, 10-bit, rotated) into a bench MP4 the board can
# decode: H.264 High, 8-bit yuv420p, fixed fps, rotation applied, no audio, one
# keyframe every FPS frames, no B-frames (clean for RTSP re-streaming).
#
# Usage: tools/host/convert_video.sh INPUT [OUTPUT]
#   OUTPUT defaults to datasets/videos/<input name>_<FPS>if.mp4 (gitignored).
#   FPS (default 18) and CRF (default 20) can be set in the environment.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VIDEOS_DIR="${SCRIPT_DIR}/../../datasets/videos"
FPS="${FPS:-18}"
CRF="${CRF:-20}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: 'ffmpeg' is required but not found in PATH." >&2
    exit 1
fi

INPUT="$1"
if [[ ! -f "$INPUT" ]]; then
    echo "Error: input not found: $INPUT" >&2
    exit 1
fi

if [[ $# -ge 2 ]]; then
    OUTPUT="$2"
else
    NAME=$(basename "$INPUT")
    OUTPUT="${VIDEOS_DIR}/${NAME%.*}_${FPS}if.mp4"
fi

# ffmpeg applies the display rotation tag itself when it re-encodes.
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" -an \
    -vf "fps=${FPS},format=yuv420p" \
    -c:v libx264 -profile:v high -preset medium -crf "$CRF" \
    -g "$FPS" -keyint_min "$FPS" -sc_threshold 0 -bf 0 \
    -movflags +faststart "$OUTPUT"

echo "wrote $OUTPUT"
ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration \
    -of default=nw=1 "$OUTPUT"
