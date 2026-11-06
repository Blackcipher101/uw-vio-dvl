import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import trajectory as tj  # noqa: E402


@pytest.fixture(scope="session")
def traj():
    return tj.build_trajectory()
