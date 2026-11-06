"""bench_sim.sh passes --vision-blackout to every blackout config (also with DVL) and writes distinct per-config outputs."""
import json
import subprocess

from conftest import SIM


def test_blackout_flag_reaches_dvl_run(tmp_path):
    bench = tmp_path / "runs" / "clear" / "bench"
    bench.mkdir(parents=True)
    (bench / "convert_info.json").write_text(json.dumps({
        "left_topic": "/l", "right_topic": "/r", "imu_topic": "/i", "pressure_topic": "/p", "suggested_blackout": "159:20"}))
    (bench / "dvl.csv").write_text("t,vx,vy,vz,valid\n")
    fake = tmp_path / "run_bag"
    fake.write_text('#!/bin/bash\nexit 1\n')
    fake.chmod(0o755)
    env = {"RUN_BAG": str(fake), "RUNS": str(tmp_path / "runs"), "REPO": str(SIM.parent), "PATH": "/usr/bin:/bin"}
    subprocess.run(["bash", str(SIM / "bench_sim.sh"), "clear_dvl", "clear_blackout", "clear_blackout_dvl"], env=env,
                   capture_output=True, text=True)
    cmd = {n: (tmp_path / "runs" / "results" / f"clear_{n}.cmd").read_text() for n in ("dvl", "blackout", "blackout_dvl")}
    assert "--vision-blackout 159:20" in cmd["blackout_dvl"] and "--dvl-file" in cmd["blackout_dvl"]
    assert "--vision-blackout" in cmd["blackout"] and "--dvl-file" not in cmd["blackout"]
    assert "--vision-blackout" not in cmd["dvl"]
