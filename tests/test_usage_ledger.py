"""The usage ledger: counts what was billed, and never breaks a run.

Cost per *accepted* scenario is the number that says whether an
acceptance-rate mechanism paid for itself, and it cannot be computed without
counting requests. The ledger is therefore load-bearing for every comparison
— but it is only bookkeeping, so a response shape it cannot read must cost a
token count, never the run.
"""

import threading

import pytest

from conflictxgame.common.model_wrappers import ModelWrapper, UsageLedger


class Usage:
    def __init__(self, prompt=None, completion=None, input_tokens=None, output_tokens=None):
        if prompt is not None:
            self.prompt_tokens = prompt
        if completion is not None:
            self.completion_tokens = completion
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens


class Response:
    def __init__(self, usage=None):
        self.usage = usage


class LedgerClient(ModelWrapper):
    """Records a successful request the way the real clients do."""

    def __init__(self, response=None, **kwargs):
        super().__init__("ledger-model", **kwargs)
        self.response = response

    def generate(self, messages):
        self._handle_api_success(self.response)
        return "ok"

    def batch_generate(self, messages_list, verbose=False):
        return self._dispatch_batch(messages_list, verbose=verbose)

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def prompts(n):
    return [[{"role": "user", "content": f"p{i}"}] for i in range(n)]


def test_every_billed_request_is_counted():
    client = LedgerClient(Response(Usage(prompt=100, completion=20)))
    client.batch_generate(prompts(5))
    snapshot = client.usage.snapshot()
    assert snapshot["requests"] == 5
    assert snapshot["prompt_tokens"] == 500
    assert snapshot["completion_tokens"] == 100
    assert snapshot["tokens_known"] == 5


def test_anthropic_style_usage_fields_are_understood():
    client = LedgerClient(Response(Usage(input_tokens=7, output_tokens=3)))
    client.generate([])
    snapshot = client.usage.snapshot()
    assert snapshot["prompt_tokens"] == 7
    assert snapshot["completion_tokens"] == 3


@pytest.mark.parametrize("response", [None, "a bare string error body", Response(None), object()])
def test_an_unreadable_response_still_counts_as_a_request(response):
    """Gateways answer with all sorts of shapes; none may abort a run."""
    client = LedgerClient(response)
    client.generate([])
    snapshot = client.usage.snapshot()
    assert snapshot["requests"] == 1
    # Counted, but honest about not knowing the tokens.
    assert snapshot["tokens_known"] == 0
    assert snapshot["prompt_tokens"] == 0


def test_partial_usage_is_recorded_as_partial():
    client = LedgerClient(Response(Usage(prompt=11)))
    client.generate([])
    snapshot = client.usage.snapshot()
    assert snapshot["prompt_tokens"] == 11
    assert snapshot["completion_tokens"] == 0
    assert snapshot["tokens_known"] == 1


def test_counting_is_safe_under_concurrency():
    """batch_generate may run generate from several threads at once."""
    client = LedgerClient(Response(Usage(prompt=1, completion=1)), max_workers=8)
    client.batch_generate(prompts(200))
    assert client.usage.snapshot()["requests"] == 200


def test_failures_are_not_counted_as_spend():
    """The circuit breaker's failures never reached a billable response."""
    client = LedgerClient(Response(Usage(prompt=1, completion=1)))
    client._handle_api_failure("boom")
    assert client.usage.snapshot()["requests"] == 0


def test_success_still_resets_the_failure_counter():
    """The ledger rides along with the retry logic; it must not displace it."""
    client = LedgerClient(Response(Usage(prompt=1, completion=1)))
    client._handle_api_failure("boom")
    assert client.consecutive_failures == 1
    client.generate([])
    assert client.consecutive_failures == 0


def test_since_reports_one_phase_of_a_run():
    ledger = UsageLedger()
    ledger.record(Response(Usage(prompt=5, completion=1)))
    mark = ledger.snapshot()
    ledger.record(Response(Usage(prompt=3, completion=2)))
    delta = ledger.since(mark)
    assert delta["requests"] == 1
    assert delta["prompt_tokens"] == 3


def test_clients_count_separately():
    """A generator and a judge on the same model id must not share a tally."""
    generator = LedgerClient(Response(Usage(prompt=1, completion=1)))
    judge = LedgerClient(Response(Usage(prompt=1, completion=1)))
    generator.batch_generate(prompts(3))
    judge.generate([])
    assert generator.usage.snapshot()["requests"] == 3
    assert judge.usage.snapshot()["requests"] == 1
