#!/bin/bash
# Run the Stonefish seabed scenario on ws1, record a rosbag2, stop everything cleanly, convert to benchmark inputs.
#
#   sim/run_sim_ws1.sh [clear|turbid|both]            (default: both)
#
# Environment (defaults for ws1):
#   SIM_WS=$HOME/uwvio/sim/ws  SIM_PREFIX=$HOME/uwvio/sim/install   Stonefish + stonefish_ros2 install
#   SIM_DISPLAY=:1             X display used for GPU rendering
#   OUT=$HOME/uwvio/sim_runs   one directory per variant: $OUT/<variant>/{raw_bag,bench,sim.log,record.log,convert.log}
#   RATE=200                   physics rate [Hz] (must equal sim/rig.py SIM_RATE_HZ: the converter relies on it)
#   CAM_RATE=20                camera rate [Hz]; if set, scenarios are regenerated (e.g. CAM_RATE=10 to cut render cost)
#   WIN_W=1000 WIN_H=750       simulator window size (smaller = cheaper)
#   QUALITY=medium             stonefish_simulator rendering_quality: low|medium|high
#   CAM_LATENCY_S=0.0          passed to convert.py --cam-latency-s
#   KEEP_RAW=0                 1: keep the (large, ~14 GB raw rgb8) raw bag after a successful conversion
#   MIN_FREE_GB=40             refuse to start with less free disk in $OUT
#   NO_CONVERT=0               1: stop after recording
# Processes are stopped by PID / process group (SIGINT to the recorder so rosbag2 writes its metadata, SIGTERM then
# SIGKILL to the simulator group); nothing uses pkill -f.
set -euo pipefail

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SIM_WS=${SIM_WS:-$HOME/uwvio/sim/ws}
SIM_PREFIX=${SIM_PREFIX:-$HOME/uwvio/sim/install}
OUT=${OUT:-$HOME/uwvio/sim_runs}
RATE=${RATE:-200}
QUALITY=${QUALITY:-medium}
WIN_W=${WIN_W:-1000}
WIN_H=${WIN_H:-750}
CAM_LATENCY_S=${CAM_LATENCY_S:-0.0}
KEEP_RAW=${KEEP_RAW:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}
NO_CONVERT=${NO_CONVERT:-0}
STORAGE=${STORAGE:-mcap}          # rosbag2 storage plugin for the raw recording (mcap or sqlite3)
REC_GRACE_S=${REC_GRACE_S:-300}   # seconds the recorder may take to flush after SIGINT before SIGKILL
VARIANTS=${1:-both}
[ "$VARIANTS" = both ] && VARIANTS="clear turbid"

export DISPLAY=${SIM_DISPLAY:-:1}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-67}              # keep the (large) image traffic away from other ROS graphs
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$SIM_WS/install/setup.bash"
set -u
export LD_LIBRARY_PATH="$SIM_PREFIX/lib:${LD_LIBRARY_PATH:-}"

TOPICS=(/sim/cam_left/image_color /sim/cam_right/image_color /sim/imu_sf /sim/pressure /sim/dvl /sim/traj_state /sim/gt_odom)
RECORD_S=$(python3 -c "import json,sys; print(json.load(open('$HERE/scenarios/trajectory_meta.json'))['record_s'])")
SIM_RATE_META=$(python3 -c "import json; print(json.load(open('$HERE/scenarios/trajectory_meta.json'))['sim_rate_hz'])")
[ "${RATE%.*}" = "${SIM_RATE_META%.*}" ] || { echo "RATE=$RATE differs from sim_rate_hz=$SIM_RATE_META in trajectory_meta.json" >&2; exit 2; }

REC_PID=""; SIM_PID=""
stop_group() {  # stop_group <pid> <first signal> <label> [grace_s]: signal the whole process group the child leads
  local pid=$1 sig=$2 label=$3 grace=${4:-15} pgid
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  if [ "$pgid" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then  # setsid did not give it its own group: never signal ours
    kill -"$sig" "$pid" 2>/dev/null || true
  else
    kill -"$sig" -- "-$pgid" 2>/dev/null || true
  fi
  for _ in $(seq 1 $((grace * 2))); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
  echo "$label (pid $pid) did not exit after SIG$sig; sending SIGKILL" >&2
  if [ "$pgid" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then kill -KILL "$pid" 2>/dev/null || true; else kill -KILL -- "-$pgid" 2>/dev/null || true; fi
}
cleanup() { stop_group "$SIM_PID" TERM simulator; stop_group "$REC_PID" INT recorder "$REC_GRACE_S"; }
trap cleanup EXIT

if [ -n "${CAM_RATE:-}" ]; then python3 "$HERE/gen_scenario.py" --cam-rate "$CAM_RATE" >/dev/null; fi
if [ ! -f "$HERE/assets/terrain.obj" ]; then
  echo "== generating assets"; python3 "$HERE/gen_assets.py"
fi
mkdir -p "$OUT"
free_gb=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "only ${free_gb} GB free in $OUT (< MIN_FREE_GB=$MIN_FREE_GB)" >&2; exit 3; }

for v in $VARIANTS; do
  run="$OUT/$v"
  [ ! -e "$run/raw_bag" ] || { echo "$run/raw_bag exists; move it away first" >&2; exit 4; }
  mkdir -p "$run"
  echo "== [$v] starting recorder"
  # mcap writes raw images faster than sqlite3 (the sqlite3 recorder dropped ~10 % of left frames at ~47 MB/s).
  setsid ros2 bag record -s "$STORAGE" --max-cache-size 500000000 -o "$run/raw_bag" "${TOPICS[@]}" >"$run/record.log" 2>&1 &
  REC_PID=$!
  sleep 5
  echo "== [$v] starting simulator (DISPLAY=$DISPLAY, ${RATE} Hz, quality $QUALITY)"
  setsid ros2 run stonefish_ros2 stonefish_simulator "$HERE" "$HERE/scenarios/seabed_$v.scn" "$RATE" "$WIN_W" "$WIN_H" "$QUALITY" >"$run/sim.log" 2>&1 &
  SIM_PID=$!
  echo "== [$v] waiting for the first trajectory message (scene load can take a minute)"
  ok=0
  for _ in $(seq 1 24); do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then echo "simulator exited early; see $run/sim.log" >&2; tail -20 "$run/sim.log" >&2; exit 5; fi
    if timeout 5 ros2 topic echo --once --field child_frame_id /sim/traj_state >/dev/null 2>&1; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || { echo "no /sim/traj_state within 2 min; see $run/sim.log" >&2; exit 6; }
  echo "== [$v] recording for ${RECORD_S} s"
  for _ in $(seq 1 "$(python3 -c "print(int($RECORD_S))")"); do
    kill -0 "$SIM_PID" 2>/dev/null || { echo "simulator died while recording; see $run/sim.log" >&2; exit 7; }
    sleep 1
  done
  echo "== [$v] stopping"
  # Simulator first (stop publishing), then give the recorder time to flush its cache before any SIGKILL.
  stop_group "$SIM_PID" TERM simulator; SIM_PID=""
  stop_group "$REC_PID" INT recorder "$REC_GRACE_S"; REC_PID=""
  if [ ! -f "$run/raw_bag/metadata.yaml" ]; then
    echo "== [$v] bag has no metadata (recorder killed?); reindexing" >&2
    ros2 bag reindex "$run/raw_bag" -s "$STORAGE" || true
  fi
  ros2 bag info "$run/raw_bag" | tee "$run/raw_bag_info.txt" | grep -E "Duration|Messages|/sim/" || true
  if [ "$NO_CONVERT" != 1 ]; then
    echo "== [$v] converting"
    python3 "$HERE/convert.py" --bag "$run/raw_bag" --out "$run/bench" --cam-latency-s "$CAM_LATENCY_S" 2>&1 | tee "$run/convert.log" | tail -25
    [ "$KEEP_RAW" = 1 ] || rm -rf "$run/raw_bag"
  fi
done
echo "== done; next: sim/bench_sim.sh"
