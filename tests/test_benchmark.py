import pytest

from benchmarks.agent_benchmark import run_benchmark


@pytest.mark.asyncio
async def test_deterministic_benchmark_passes():
    report = await run_benchmark()

    assert report["case_count"] == 10
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0
    assert report["total_tokens"] > 0
