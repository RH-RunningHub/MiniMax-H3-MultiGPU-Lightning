#!/usr/bin/env bash
# Submit, poll and download one 15-second T2VA request.
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:30010}"
OUT="${OUT:-h3-15s.mp4}"
REQUEST_FILE="$(mktemp)"
trap 'rm -f "$REQUEST_FILE"' EXIT

cat >"$REQUEST_FILE" <<JSON
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "t2va",
  "prompt": "A cyberpunk street musician plays a neon-lit saxophone in the rain, cinematic tracking shot, shallow depth of field.",
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 15.0
  },
  "num_inference_steps": 20,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": ${SEED:-42}
}
JSON

curl --fail --show-error --silent "$BASE/health" >/dev/null
JOB_ID="$(curl --fail --show-error --silent -X POST "$BASE/v1/videos" \
  -H 'Content-Type: application/json' --data-binary @"$REQUEST_FILE" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
echo "job: $JOB_ID"

while true; do
  RESPONSE="$(curl --fail --show-error --silent "$BASE/v1/videos/$JOB_ID")"
  STATUS="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"$RESPONSE")"
  echo "$RESPONSE"
  case "$STATUS" in
    completed) break ;;
    failed|error|cancelled|deleted) exit 1 ;;
  esac
  sleep 3
done

curl --fail --show-error --silent "$BASE/v1/videos/$JOB_ID/content" -o "$OUT"
ffprobe -v error -show_entries stream=codec_type,codec_name -of json "$OUT"
echo "saved: $OUT"
