"""Concurrency in batch_generate must change only speed, never results."""

import threading
import time

import pytest

from conflictxgame.common.model_wrappers import ModelWrapper


class RecordingClient(ModelWrapper):
    """Minimal client that records call order and simulates network latency."""

    def __init__(self, delay=0.0, **kwargs):
        super().__init__("test-model", **kwargs)
        self.delay = delay
        self.peak_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def generate(self, messages):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return messages[-1]["content"].upper()
        finally:
            with self._lock:
                self._in_flight -= 1

    def batch_generate(self, messages_list, verbose=False):
        return self._dispatch_batch(messages_list, verbose=verbose)

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def prompts(n):
    return [[{"role": "user", "content": f"p{i}"}] for i in range(n)]


def test_defaults_to_sequential():
    client = RecordingClient()
    assert client.max_workers == 1
    client.batch_generate(prompts(5))
    assert client.peak_in_flight == 1


def test_results_stay_in_input_order_when_concurrent():
    """Callers zip results against prompts, so order cannot depend on timing."""
    client = RecordingClient(max_workers=4)
    result = client.batch_generate(prompts(12))
    assert result == [f"P{i}" for i in range(12)]


def test_concurrency_actually_overlaps_requests():
    client = RecordingClient(delay=0.05, max_workers=4)
    client.batch_generate(prompts(8))
    assert client.peak_in_flight > 1


def test_single_prompt_never_spawns_a_pool():
    client = RecordingClient(max_workers=8)
    assert client.batch_generate(prompts(1)) == ["P0"]
    assert client.peak_in_flight == 1


@pytest.mark.parametrize("value", [0, -3])
def test_nonsensical_worker_counts_fall_back_to_sequential(value):
    assert RecordingClient(max_workers=value).max_workers == 1


def test_exceptions_propagate_rather_than_being_swallowed():
    """A tripped circuit breaker must still stop the run under concurrency."""

    class Failing(RecordingClient):
        def generate(self, messages):
            raise RuntimeError("circuit breaker")

    with pytest.raises(RuntimeError, match="circuit breaker"):
        Failing(max_workers=4).batch_generate(prompts(4))
