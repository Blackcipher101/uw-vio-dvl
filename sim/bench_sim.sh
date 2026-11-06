#!/bin/bash
# Run uw_ct_vio run_bag on the converted Stonefish runs and score every configuration against the exact ground truth.
#
#   sim/bench_sim.sh [config ...]        configs (default: all):
#     clear_vision        stereo-inertial-pressure, no DVL
#     clear_dvl           + DVL (physically simulated, with the natural bottom-lock loss)
#     clear_blackout      20 s camera blackout, no DVL
#     clear_blackout_dvl  20 s camera blackout + DVL
#     turbid_vision       turbid water, no DVL
#     turbid_dvl          turbid water + DVL
#
# Environment:
#   RUN_BAG=$HOME/uwvio/uw-vslam-bench/build_dvl/build/uw_ct_vio/run_bag
#   RUNS=$HOME/uwvio/sim_runs        run_sim_ws1.sh output (variant dirs with bench/)
#   REPO=$HOME/uwvio/uw-vio-dvl      for scripts/evaluate.py
#   OUT=$RUNS/results                est trajectories, logs and results.tsv
#   BASE_FLAGS                       extra run_bag flags for every run (default: the NTNU-tuned stereo/LCN set)
#   BLACKOUT=t0:dur                  default from the run's convert_info.json (20 s in the middle of leg 2)
# GT is the LEFT camera pose in world NED (exported by convert.py), so evaluate.py is used without --hand-eye.
set -uo pipefail

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
RUN_BAG=${RUN_BAG:-$HOME/uwvio/uw-vslam-bench/build_dvl/build/uw_ct_vio/run_bag}
RUNS=${RUNS:-$HOME/uwvio/sim_runs}
REPO=${REPO:-$HOME/uwvio/uw-vio-dvl}
OUT=${OUT:-$RUNS/results}
BASE_FLAGS=${BASE_FLAGS:---lcn 2 --lcn-stereo 0 --stereo-from-intrinsics 1 --huber-k 0.5}
CALIB=$HERE/calib
[ -x "$RUN_BAG" ] || { echo "run_bag not found/executable: $RUN_BAG" >&2; exit 2; }
mkdir -p "$OUT"
TSV=$OUT/results.tsv
COLS="variant\tconfig\tstatus\tate_se3_m\tate_sim3_m\tsim3_scale\trpe_trans_pct_5m\trpe_rot_deg_per_m_5m\tcoverage\tmatched\trun_s\tdvl_received\tdvl_invalid\tdvl_gated\tdvl_factors\tdvl_relocks\tframes_without_vision\tblackout_frames\tnote"
printf "$COLS\n" >"$TSV"
echo -e "$COLS" | tr '\t' ' '

json_get() {  # json_get <file> <key>   -> value or "nan"
  python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v = next(iter(d.values())).get(sys.argv[2], "nan")
print(f"{v:.6g}" if isinstance(v, float) else v)
PY
}
counter() {  # counter <log> <name>  -> integer after 'name ' in the final 'done:' line, or NA
  grep -E "^done:" "$1" | tail -1 | grep -oE "$2 [0-9.]+" | awk '{print $2}' | head -1 || true
}

run_one() {  # run_one <variant> <name> <dvl 0|1> <blackout 0|1>
  local variant=$1 name=$2 dvl=$3 blk=$4
  local bench=$RUNS/$variant/bench tag=${variant}_${name}
  local info=$bench/convert_info.json
  [ -f "$info" ] || { printf "$variant\t$name\tno-bench-data\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t$bench\n" | tee -a "$TSV"; return; }
  read -r LT RT IT PT BO < <(python3 -c "
import json; d=json.load(open('$info'))
print(d['left_topic'], d['right_topic'], d['imu_topic'], d['pressure_topic'], d['suggested_blackout'])")
  local flags="$BASE_FLAGS"
  [ "$dvl" = 1 ] && flags="$flags --dvl-file $bench/dvl.csv --dvl-extrinsic $CALIB/dvl_extrinsic.yaml"
  [ "$blk" = 1 ] && flags="$flags --vision-blackout ${BLACKOUT:-$BO}"
  local t0=$SECONDS status=ok note=""
  echo "$RUN_BAG --bag $bench/bench_bag $flags --out $OUT/$tag.tum" >"$OUT/$tag.cmd"  # exact flags of this run
  # shellcheck disable=SC2086  # flags are intentionally word-split
  "$RUN_BAG" --bag "$bench/bench_bag" --intrinsics "$CALIB/camchain.yaml" --imucam "$CALIB/imucam.yaml" \
      --imu-yaml "$CALIB/imu.yaml" --left cam0 --right cam1 --left-topic "$LT" --right-topic "$RT" \
      --imu-topic "$IT" --pressure-topic "$PT" --water-density 1000 $flags \
      --out "$OUT/$tag.tum" >"$OUT/$tag.log" 2>&1 || { status=run_bag-failed; note=$(tail -1 "$OUT/$tag.log" | tr '\t' ' '); }
  local dur=$((SECONDS - t0))
  if [ "$status" = ok ]; then
    python3 "$REPO/scripts/evaluate.py" "$bench/gt_leftcam.tum" "$OUT/$tag.tum" --json "$OUT/$tag.json" >"$OUT/$tag.eval.log" 2>&1 \
      || { status=evaluate-failed; note=$(tail -1 "$OUT/$tag.eval.log" | tr '\t' ' '); }
  fi
  if [ "$status" = ok ] && grep -q '"error"' "$OUT/$tag.json" 2>/dev/null; then status=scored-with-error; note=$(json_get "$OUT/$tag.json" error); fi
  if [ "$status" = ok ] || [ "$status" = scored-with-error ]; then
    local g=("ate_se3_m" "ate_sim3_m" "sim3_scale" "rpe_trans_pct_5m" "rpe_rot_deg_per_m_5m" "coverage" "matched")
    local vals=() k
    for k in "${g[@]}"; do vals+=("$(json_get "$OUT/$tag.json" "$k")"); done
  else
    local vals=(nan nan nan nan nan nan nan)
  fi
  local nv; nv=$(grep -E "^done:" "$OUT/$tag.log" | tail -1 | sed -E 's/.*frames, ([0-9]+) without vision.*/\1/')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$variant" "$name" "$status" \
    "${vals[0]}" "${vals[1]}" "${vals[2]}" "${vals[3]}" "${vals[4]}" "${vals[5]}" "${vals[6]}" "$dur" \
    "$(counter "$OUT/$tag.log" dvl_received)" "$(counter "$OUT/$tag.log" dvl_invalid)" "$(counter "$OUT/$tag.log" dvl_gated)" \
    "$(counter "$OUT/$tag.log" dvl_factors)" "$(counter "$OUT/$tag.log" dvl_relocks)" "$nv" "$(counter "$OUT/$tag.log" blackout_frames)" "$note" | tee -a "$TSV"
}

declare -A CFG=(
  [clear_vision]="clear vision 0 0"          [clear_dvl]="clear dvl 1 0"
  [clear_blackout]="clear blackout 0 1"      [clear_blackout_dvl]="clear blackout_dvl 1 1"
  [turbid_vision]="turbid vision 0 0"        [turbid_dvl]="turbid dvl 1 0")
ORDER=(clear_vision clear_dvl clear_blackout clear_blackout_dvl turbid_vision turbid_dvl)
SEL=("$@"); [ ${#SEL[@]} -gt 0 ] || SEL=("${ORDER[@]}")
for c in "${SEL[@]}"; do
  [ -n "${CFG[$c]:-}" ] || { echo "unknown config $c (valid: ${ORDER[*]})" >&2; exit 2; }
  # shellcheck disable=SC2086
  run_one ${CFG[$c]}
done
echo "results: $TSV"
