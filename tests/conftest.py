import pytest

from beavr_bench.sim import list_available_scenes


@pytest.fixture(scope="session")
def available_scenes():
    """Returns a list of all baked scene names."""
    return list_available_scenes()
