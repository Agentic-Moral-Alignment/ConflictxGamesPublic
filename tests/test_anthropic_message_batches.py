"""Focused coverage for the opt-in Anthropic Message Batches transport."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from conflictxgame.common import model_wrappers
from conflictxgame.common.model_wrappers import (
    AnthropicClient,
    MessageBatchStateError,
)


def message(text, input_tokens=1, output_tokens=1, *, thinking=False):
    blocks = []
    if thinking:
        blocks.append(SimpleNamespace(type="thinking", text=None))
    blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def batch_item(index, result):
    return SimpleNamespace(custom_id=f"request_{index:06d}", result=result)


def succeeded(index, text, input_tokens=1, output_tokens=1, *, thinking=False):
    return batch_item(
        index,
        SimpleNamespace(
            type="succeeded",
            message=message(
                text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking=thinking,
            ),
        ),
    )


def errored(index, error_type="invalid_request_error", detail="bad request"):
    error = SimpleNamespace(
        error=SimpleNamespace(type=error_type, message=detail),
        type="error",
    )
    return batch_item(index, SimpleNamespace(type="errored", error=error))


def terminal(index, result_type):
    return batch_item(index, SimpleNamespace(type=result_type))


class FakeBatches:
    def __init__(self, results=(), create_status="ended", retrieve_events=()):
        self.result_items = list(results)
        self.create_status = create_status
        self.retrieve_events = list(retrieve_events)
        self.create_calls = []
        self.retrieve_calls = []
        self.results_calls = []

    def create(self, *, requests):
        self.create_calls.append(requests)
        return SimpleNamespace(
            id="msgbatch_test",
            processing_status=self.create_status,
        )

    def retrieve(self, batch_id):
        self.retrieve_calls.append(batch_id)
        event = self.retrieve_events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return SimpleNamespace(id=batch_id, processing_status=event)

    def results(self, batch_id):
        self.results_calls.append(batch_id)
        return iter(self.result_items)


class FakeMessages:
    def __init__(self, batches, sync_responses=()):
        self.batches = batches
        self.sync_responses = list(sync_responses)
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.sync_responses.pop(0)


class FakeAnthropic:
    def __init__(self, batches, sync_responses=()):
        self.messages = FakeMessages(batches, sync_responses=sync_responses)


def make_client(monkeypatch, service, **kwargs):
    monkeypatch.setattr(model_wrappers, "Anthropic", lambda: service)
    return AnthropicClient("claude-test", max_retries=1, **kwargs)


def prompts(count):
    return [
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": f"Prompt {index}"},
        ]
        for index in range(count)
    ]


def test_message_batches_are_explicit_and_require_durable_state(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="batch_state_dir"):
        AnthropicClient("claude-test", use_message_batches=True)

    batches = FakeBatches(results=[succeeded(0, "ok")])
    service = FakeAnthropic(batches)
    monkeypatch.setattr(model_wrappers, "Anthropic", lambda: service)
    client = AnthropicClient("claude-test", max_retries=1)
    # Construction and the default batch path retain their historical sync
    # behavior; no manifest key or directory is required.
    client.client = service
    service.messages.sync_responses.append(message("sync"))
    assert client.batch_generate(prompts(1)) == ["sync"]
    assert not batches.create_calls


def test_batch_params_match_the_standard_messages_request(monkeypatch, tmp_path):
    kwargs = {
        "temperature": 0.4,
        "max_tokens": 321,
        "top_p": 0.8,
        "stop_sequences": ["END"],
    }
    sync_service = FakeAnthropic(
        FakeBatches(), sync_responses=[message("sync", 3, 4, thinking=True)]
    )
    sync_client = make_client(monkeypatch, sync_service, **kwargs)
    assert sync_client.generate(prompts(1)[0]) == "sync"

    batch_api = FakeBatches(results=[succeeded(0, "batch", 3, 4, thinking=True)])
    batch_service = FakeAnthropic(batch_api)
    batch_client = make_client(
        monkeypatch,
        batch_service,
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
        **kwargs,
    )
    assert batch_client.batch_generate(prompts(1), batch_key="parity") == ["batch"]

    assert batch_api.create_calls[0][0]["params"] == sync_service.messages.create_calls[0]
    assert batch_api.create_calls[0][0]["custom_id"] == "request_000000"
    assert batch_client.usage.snapshot() == {
        "requests": 1,
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "tokens_known": 1,
    }


def test_results_are_reordered_and_mixed_terminal_states_are_preserved(
    monkeypatch, tmp_path, caplog
):
    batch_api = FakeBatches(
        results=[
            terminal(4, "canceled"),
            succeeded(3, "fourth", 7, 8),
            errored(2),
            succeeded(0, "first", 2, 3),
            terminal(1, "expired"),
        ]
    )
    client = make_client(
        monkeypatch,
        FakeAnthropic(batch_api),
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )

    assert client.batch_generate(prompts(5), batch_key="mixed") == [
        "first", "", "", "fourth", ""
    ]
    assert client.usage.snapshot() == {
        "requests": 2,
        "prompt_tokens": 9,
        "completion_tokens": 11,
        "tokens_known": 2,
    }
    assert "invalid_request_error" in caplog.text


def test_in_progress_manifest_resumes_without_resubmitting(monkeypatch, tmp_path):
    batch_api = FakeBatches(
        results=[succeeded(0, "done", 5, 6)],
        create_status="in_progress",
        retrieve_events=[KeyboardInterrupt(), "ended"],
    )
    service = FakeAnthropic(batch_api)
    first = make_client(
        monkeypatch,
        service,
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )
    with pytest.raises(KeyboardInterrupt):
        first.batch_generate(prompts(1), batch_key="resume-me")

    manifests = list(Path(tmp_path).glob("*.json"))
    assert len(manifests) == 1
    assert len(batch_api.create_calls) == 1

    second = make_client(
        monkeypatch,
        service,
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )
    assert second.batch_generate(prompts(1), batch_key="resume-me") == ["done"]
    assert len(batch_api.create_calls) == 1
    assert batch_api.retrieve_calls == ["msgbatch_test", "msgbatch_test"]
    assert second.usage.snapshot()["requests"] == 1


def test_completed_manifest_is_a_local_result_cache_and_accounts_once(
    monkeypatch, tmp_path
):
    batch_api = FakeBatches(
        results=[succeeded(1, "second", 4, 5), succeeded(0, "first", 2, 3)]
    )
    service = FakeAnthropic(batch_api)
    first = make_client(
        monkeypatch,
        service,
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )
    assert first.batch_generate(prompts(2), batch_key="cached") == ["first", "second"]

    second = make_client(
        monkeypatch,
        service,
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )
    assert second.batch_generate(prompts(2), batch_key="cached") == ["first", "second"]
    assert second.batch_generate(prompts(2), batch_key="cached") == ["first", "second"]
    assert len(batch_api.create_calls) == 1
    assert len(batch_api.results_calls) == 1
    assert second.usage.snapshot() == {
        "requests": 2,
        "prompt_tokens": 6,
        "completion_tokens": 8,
        "tokens_known": 2,
    }


def _overwrite_completed_manifest(client, batch_key, request_count, records):
    requests = [
        {
            'custom_id': f'request_{index:06d}',
            'params': client._anthropic_request_params(prompt),
        }
        for index, prompt in enumerate(prompts(request_count))
    ]
    path = client._batch_manifest_path(batch_key)
    AnthropicClient._write_batch_manifest(path, {
        'version': client._BATCH_MANIFEST_VERSION,
        'batch_key': batch_key,
        'fingerprint': client._batch_fingerprint(requests),
        'request_count': request_count,
        'batch_id': 'msgbatch_test',
        'status': 'ended',
        'results': records,
    })


def test_completed_manifest_must_contain_every_result(monkeypatch, tmp_path):
    client = make_client(
        monkeypatch, FakeAnthropic(FakeBatches()), use_message_batches=True,
        batch_state_dir=str(tmp_path), batch_poll_interval=0,
    )
    _overwrite_completed_manifest(client, 'short', 3, [
        {'custom_id': 'request_000000', 'type': 'succeeded', 'text': 'first', 'usage': {}},
    ])
    with pytest.raises(MessageBatchStateError, match='ended without results'):
        client.batch_generate(prompts(3), batch_key='short')


def test_completed_manifest_is_reordered_to_request_order(monkeypatch, tmp_path):
    client = make_client(
        monkeypatch, FakeAnthropic(FakeBatches()), use_message_batches=True,
        batch_state_dir=str(tmp_path), batch_poll_interval=0,
    )
    _overwrite_completed_manifest(client, 'permuted', 3, [
        {'custom_id': 'request_000002', 'type': 'succeeded', 'text': 'third', 'usage': {}},
        {'custom_id': 'request_000000', 'type': 'succeeded', 'text': 'first', 'usage': {}},
        {'custom_id': 'request_000001', 'type': 'succeeded', 'text': 'second', 'usage': {}},
    ])
    assert client.batch_generate(prompts(3), batch_key='permuted') == [
        'first', 'second', 'third',
    ]


def test_completed_manifest_rejects_null_text(monkeypatch, tmp_path):
    client = make_client(
        monkeypatch, FakeAnthropic(FakeBatches()), use_message_batches=True,
        batch_state_dir=str(tmp_path), batch_poll_interval=0,
    )
    _overwrite_completed_manifest(client, 'null-text', 1, [
        {'custom_id': 'request_000000', 'type': 'succeeded', 'text': None, 'usage': {}},
    ])
    with pytest.raises(MessageBatchStateError, match='non-string text'):
        client.batch_generate(prompts(1), batch_key='null-text')


def test_reusing_key_with_changed_requests_is_refused(monkeypatch, tmp_path):
    batch_api = FakeBatches(results=[succeeded(0, "original")])
    client = make_client(
        monkeypatch,
        FakeAnthropic(batch_api),
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )
    assert client.batch_generate(prompts(1), batch_key="stable") == ["original"]

    changed = [[{"role": "user", "content": "Different prompt"}]]
    with pytest.raises(MessageBatchStateError, match="reused with different"):
        client.batch_generate(changed, batch_key="stable")
    assert len(batch_api.create_calls) == 1


def test_polling_waits_until_ended(monkeypatch, tmp_path):
    batch_api = FakeBatches(
        results=[succeeded(0, "eventually")],
        create_status="in_progress",
        retrieve_events=["in_progress", "canceling", "ended"],
    )
    client = make_client(
        monkeypatch,
        FakeAnthropic(batch_api),
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
        batch_poll_interval=0,
    )
    assert client.batch_generate(prompts(1), batch_key="poll") == ["eventually"]
    assert len(batch_api.retrieve_calls) == 3


def test_batch_mode_requires_a_caller_key(monkeypatch, tmp_path):
    client = make_client(
        monkeypatch,
        FakeAnthropic(FakeBatches()),
        use_message_batches=True,
        batch_state_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="batch_key"):
        client.batch_generate(prompts(1))
