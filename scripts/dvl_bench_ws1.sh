#!/bin/bash
# Sim-DVL benchmark on the 6 NTNU trajectories (run on ws1; needs python3 + numpy/scipy/pyyaml).
# For each trajectory: simulate DVL CSVs for every scenario from the mocap, then run
# ~/uwvio/ntnu_all.sh once per tag (it substitutes {T} with the trajectory number 1..6).
#   scripts/dvl_bench_ws1.sh [tag ...]        # default: all tags below, in order
#   SKIP_SIM=1 scripts/dvl_bench_ws1.sh dvl_full   # reuse CSVs already in $OUT
# Every result is "sim-DVL": it proves plumbing and dropout behaviour, not real-sensor accuracy.
set -euo pipefail

# Base run_bag flags = tag h05 (swgw + huber 0.5), verified by reproduction on ws1.
BASE_FLAGS="--lcn 2 --lcn-stereo 0 --stereo-from-intrinsics 1 --imu-yaml $HOME/uwvio/configs_dev/imu_gw1e-2.yaml --huber-k 0.5${EXTRA_FLAGS:+ $EXTRA_FLAGS}"  # = tag h05 (reproduced 2026-11-04: 0.401 m / 6.75 %)

UWVIO=${UWVIO:-$HOME/uwvio}
RUNNER=${RUNNER:-$UWVIO/ntnu_all.sh}
DATA=${DATA:-$HOME/uw_datasets/ntnu/subset-3-mocap-gt}
# Kalibr imucam yaml of the above-water extrinsics (verify the path on ws1; NTNU left camera = cam1).
IMUCAM=${IMUCAM:-$UWVIO/calibration/stereo_extrinsics_above_water/camchain-imucam-ariel_stereo_extrinsics_above_water.yaml}
REF_TAG=${REF_TAG:-swgw}                       # reference run for the hand-eye X (T_body_cam)
OUT=${OUT:-$UWVIO/results/dvl_sim}
SIM=${SIM:-$(dirname "$(readlink -f "$0")")/sim_dvl.py}
BLACKOUT=${BLACKOUT:-60:20}                    # --vision-blackout t0:dur [s after the first image]
SEED=${SEED:-1}
TRAJ=${TRAJ:-"1 2 3 4 5 6"}

ALL_TAGS="dvl_none_off dvl_full dvl_periodic dvl_long dvl_bursts dvl_outliers blackout20_nodvl blackout20_dvl"
TAGS=${*:-$ALL_TAGS}

declare -A SCENARIO=([dvl_full]=none [dvl_periodic]=periodic [dvl_long]=long [dvl_bursts]=bursts
                     [dvl_outliers]=outliers [blackout20_dvl]=none)

mkdir -p "$OUT"
if [ -z "${SKIP_SIM:-}" ]; then
  for T in $TRAJ; do
    d=$OUT/traj_0$T; mkdir -p "$d"
    for sc in none periodic long bursts outliers; do
      python3 "$SIM" "$DATA/traj_0$T/traj_0${T}_mocap.tum" \
        --hand-eye-est "$UWVIO/results/ntnu_all/$REF_TAG/traj_0$T.tum" --imucam "$IMUCAM" --cam cam1 \
        --scenario $sc --seed "$SEED" --out "$d/$sc.csv"
    done
  done
fi

# {T} is substituted by the runner with 01..06; the CSV/yaml paths follow $OUT/traj_{T}/<scenario>.{csv,yaml}.
dvl_flags() { echo "--dvl-file $OUT/traj_{T}/$1.csv --dvl-extrinsic $OUT/traj_{T}/$1.yaml"; }

for tag in $TAGS; do
  case $tag in
    dvl_none_off)   flags="$BASE_FLAGS" ;;
    blackout20_nodvl) flags="$BASE_FLAGS --vision-blackout $BLACKOUT" ;;
    blackout20_dvl) flags="$BASE_FLAGS --vision-blackout $BLACKOUT $(dvl_flags none)" ;;
    dvl_full|dvl_periodic|dvl_long|dvl_bursts|dvl_outliers) flags="$BASE_FLAGS $(dvl_flags "${SCENARIO[$tag]}")" ;;
    *) echo "unknown tag $tag (valid: $ALL_TAGS)" >&2; exit 2 ;;
  esac
  echo "=== $tag: $RUNNER $tag $flags"
  # shellcheck disable=SC2086  # flags are intentionally word-split
  "$RUNNER" "$tag" $flags
done
