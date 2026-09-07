#!/bin/bash
# Submit -> poll -> download example for the MiniMax-H3 sglang video API.
# Usage: BASE=http://127.0.0.1:30010 bash api_example.sh
set -e
BASE=${BASE:-http://127.0.0.1:30010}

cat > /tmp/h3_req.json <<'JSON'
{
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": "a streamer dancing with big swings in her live room",
  "seconds": 15,
  "task": "t2va",
  "conditions": [],
  "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 15.0},
  "num_inference_steps": 4,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 20260904
}
JSON

echo "== submit =="
TASK_ID=$(curl -s -X POST "$BASE/v1/videos" -H 'Content-Type: application/json' \
  -d @/tmp/h3_req.json | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
echo "task: $TASK_ID"

echo "== poll =="
while :; do
  S=$(curl -s "$BASE/v1/videos/$TASK_ID" | python3 -c \
    "import sys,json;d=json.load(sys.stdin);print(d['status'],d.get('inference_time_s'),d.get('peak_memory_mb'))")
  echo "$S"
  case "$S" in completed*|success*) break;; failed*|error*) exit 1;; esac
  sleep 5
done

echo "== download =="
curl -s -o h3_out.mp4 "$BASE/v1/videos/$TASK_ID/content"
ls -l h3_out.mp4
echo "DONE -> h3_out.mp4"
