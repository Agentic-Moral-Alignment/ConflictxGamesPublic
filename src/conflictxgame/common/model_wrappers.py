"""Shared LLM client layer for both ConflictxGame pipelines.

This module provides a single interface (:class:`ModelWrapper`) over three
backends — the OpenAI Chat Completions API (:class:`OpenAIClient`), the
Anthropic Messages API (:class:`AnthropicClient`), and locally hosted
open-weight models via vLLM (:class:`VLLMClient`). Both the solo (dyadic)
pipeline and the game-theoretic pipeline in :mod:`conflictxgame.games`
construct their models through :meth:`ModelWrapper.create`.

Provider routing:
    :meth:`ModelWrapper.create` first honors an explicit ``provider:`` prefix
    on the model name (one of :data:`_PROVIDER_PREFIXES`, e.g.
    ``openai:anthropic.claude-sonnet-5``); the prefix forces the client class
    and is stripped before the id is sent to the API. Without a prefix, the
    provider is inferred by substring: names containing ``gpt`` go to OpenAI,
    names containing ``claude`` go to Anthropic, and anything else is loaded
    locally with vLLM.

OpenAI-compatible gateways:
    :class:`OpenAIClient` constructs ``OpenAI()`` with no arguments, so the
    SDK reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the
    environment. Pointing ``OPENAI_BASE_URL`` at an OpenAI-compatible gateway
    (the institutional gateway, LiteLLM, Open WebUI, ...) and using the ``openai:``
    prefix therefore lets any model the gateway serves — including
    ``anthropic.claude-*`` ids — be driven through the OpenAI client.

Lazy SDK imports:
    The ``openai``, ``anthropic``, and ``vllm`` packages are imported inside
    ``try``/``except`` blocks at module load, with placeholders installed
    when missing. This module (and everything that only needs the
    :class:`ModelWrapper` interface) is thus importable on machines where a
    given backend is not installed; each client raises an informative
    ``ImportError`` at construction time if its backend is unavailable.

Provenance: adapted from ConflictScope's ``src/model_wrappers.py`` (MIT,
(c) 2025 Andy Liu): https://github.com/andyjliu/conflictscope/blob/main/src/model_wrappers.py
Added here: explicit ``provider:`` routing, OpenAI-compatible gateway
handling, and filename-safe short model names. See ``ATTRIBUTION.md``.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import threading
import time
from types import SimpleNamespace
from typing import List, Optional, Dict, TypedDict, Any
import logging
import re
import math
import numpy as np
from collections import Counter
from tqdm import tqdm

from conflictxgame.common.portable_lock import replace_with_retry
from conflictxgame.common.reasoning_capture import capture_response

# Provider SDKs are imported lazily so that this module (and the modules that
# only need the ModelWrapper interface, e.g. the game-theoretic pipeline) can be
# imported on machines where a given backend is not installed. Each client class
# raises an informative ImportError at construction time if its backend is missing.
try:
    from openai import OpenAI, APIError
except ImportError:
    OpenAI = None

    class APIError(Exception):
        """Placeholder used when the ``openai`` package is not installed."""

try:
    from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError
except ImportError:
    Anthropic = None

    class APIConnectionError(Exception):
        """Placeholder used when the ``anthropic`` package is not installed."""

    class RateLimitError(Exception):
        """Placeholder used when the ``anthropic`` package is not installed."""

    class APIStatusError(Exception):
        """Placeholder used when the ``anthropic`` package is not installed."""

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

#: Module-level logger; clients log retries, parameter fallbacks, and failures here.
logger = logging.getLogger(__name__)


class EmptyResponseError(Exception):
    """A model returned empty, null, or malformed content (e.g. a reasoning
    model that spent its whole completion budget on reasoning tokens, or a
    gateway that answered with an unparsed error body). Treated as retryable
    by the client wrappers."""


class MessageBatchStateError(RuntimeError):
    """A persisted Anthropic Message Batch cannot be resumed safely.

    This is deliberately distinct from an API error.  It means a caller reused
    a stable ``batch_key`` for different requests, or a local manifest is
    malformed, and submitting anything automatically could duplicate paid
    work or attach responses to the wrong prompts.
    """


def _records_in_request_order(
    records: List[dict], requests: List[dict], batch_id: str
) -> List[dict]:
    """Validate persisted batch records and restore provider request order."""
    expected_ids = {request['custom_id'] for request in requests}
    by_custom_id: Dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise MessageBatchStateError(
                f"Anthropic batch {batch_id} contains a non-object result record"
            )
        custom_id = record.get('custom_id')
        if custom_id not in expected_ids:
            raise MessageBatchStateError(
                f"Anthropic batch {batch_id} returned unknown custom_id {custom_id!r}"
            )
        if custom_id in by_custom_id:
            raise MessageBatchStateError(
                f"Anthropic batch {batch_id} returned duplicate custom_id {custom_id!r}"
            )
        if not isinstance(record.get('text'), str):
            raise MessageBatchStateError(
                f"Anthropic batch {batch_id} result {custom_id!r} has non-string text"
            )
        by_custom_id[custom_id] = record

    missing_ids = expected_ids.difference(by_custom_id)
    if missing_ids:
        raise MessageBatchStateError(
            f"Anthropic batch {batch_id} ended without results for {sorted(missing_ids)}"
        )
    return [by_custom_id[request['custom_id']] for request in requests]


def completion_text(response) -> str:
    """Extract the assistant's text from a chat-completion response.

    OpenAI-compatible gateways (LiteLLM in front of Vertex-served Claude, as
    on the gateway) intermittently answer with a **bare string** — an error body
    that never parsed into a ``ChatCompletion`` — instead of the expected
    object. Reaching into ``.choices`` then raises ``AttributeError``, which
    is not one of the exceptions the retry loops catch, so a single bad
    response would abort an entire evaluation run.

    Any unexpected response shape is therefore converted into a retryable
    :class:`EmptyResponseError`.

    Args:
        response: Whatever the SDK returned from ``chat.completions.create``.

    Returns:
        The completion text.

    Raises:
        EmptyResponseError: If the response is a bare string, is missing the
            expected fields, or carries empty content.
    """
    if isinstance(response, str):
        raise EmptyResponseError(
            f"API returned a bare string instead of a completion: {response[:200]!r}"
        )
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        raise EmptyResponseError(
            f"Malformed completion response ({type(response).__name__}): {e}"
        ) from e
    if not content:
        # Reasoning models can return null content when the completion budget
        # is consumed by reasoning tokens.
        raise EmptyResponseError('No content returned from model')
    return content


def gpus_needed(model_name: str, bytes_per_param: int = 2, overhead_factor: float = 0.2, gpu_memory_gb: int = 48) -> int:
    """Estimate the number of GPUs needed to host a model locally with vLLM.

    API-served models (names containing ``claude`` or ``gpt``) need no local
    GPUs. For open-weight models the parameter count is parsed from the model
    name (the first ``<digits>b`` group, e.g. ``"8b"`` in
    ``"meta-llama/Llama-3-8b"``), converted to a memory estimate of
    ``params * bytes_per_param * (1 + overhead_factor)`` GB, and divided by
    the per-GPU memory. Used to set vLLM's ``tensor_parallel_size`` in
    :class:`VLLMClient`.

    Args:
        model_name: Model name or path; parameter size is inferred from it.
        bytes_per_param: Bytes per parameter (2 for fp16/bf16 weights).
        overhead_factor: Fractional memory overhead for activations/KV cache.
        gpu_memory_gb: Memory per GPU in GB.

    Returns:
        0 for API-served models; otherwise the estimated GPU count (minimum 1,
        also the fallback when no parameter size can be parsed from the name).
    """
    if 'claude' in model_name.lower() or 'gpt' in model_name.lower():
        return 0
    
    match = re.search(r'(\d+)b', model_name.lower())
    if not match:
        print(f"Could not extract parameter size from model name: {model_name}. Defaulting to 1 GPU.")
        return 1
    
    param_size_billions = int(match.group(1))
    memory_gb = param_size_billions * bytes_per_param * (1 + overhead_factor)
    gpus = math.ceil(memory_gb / gpu_memory_gb)
    return max(1, gpus)

def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, ignoring unusable values.

    A malformed setting should not abort a long run, so anything that is not
    a positive integer falls back to the default with a warning.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%d is below 1; using %d", name, value, default)
        return default
    return value


class UsageLedger:
    """Thread-safe tally of what a client actually spent.

    Nothing in the pipelines measured cost before this: a client counted only
    its consecutive failures, so "did the acceptance-rate mechanism pay for
    itself" could not be answered — the interesting number is calls (and
    tokens) per *accepted* scenario, not the acceptance rate on its own.

    One ledger lives on each :class:`ModelWrapper`, so a generator and a judge
    built from the same model id are still counted separately. Recording
    happens in :meth:`ModelWrapper._handle_api_success`, i.e. once per billed
    request; retried attempts that ended in an error are not counted, matching
    what a provider bills for a failed call.

    Token counts are best-effort: not every gateway returns a ``usage`` block,
    and :attr:`tokens_known` reports how many of the recorded requests did, so
    a partial count is never mistaken for a complete one.

    The lock matters because :meth:`ModelWrapper._dispatch_batch` may call
    :meth:`generate` from several threads at once.
    """

    def __init__(self):
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        #: Requests whose response carried a usable ``usage`` block.
        self.tokens_known = 0
        self._lock = threading.Lock()

    def record(self, response=None) -> None:
        """Count one billed request, and its tokens when the response has them.

        Never raises: a ledger is bookkeeping, and gateways answer with all
        sorts of shapes (including bare strings — see
        :func:`completion_text`). A response it cannot read is counted as a
        request with unknown tokens.
        """
        prompt = completion = None
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt = getattr(usage, "prompt_tokens", None)
                if prompt is None:
                    prompt = getattr(usage, "input_tokens", None)  # Anthropic
                completion = getattr(usage, "completion_tokens", None)
                if completion is None:
                    completion = getattr(usage, "output_tokens", None)
        except Exception:  # noqa: BLE001 - accounting must never break a run
            prompt = completion = None
        with self._lock:
            self.requests += 1
            if isinstance(prompt, int) or isinstance(completion, int):
                self.tokens_known += 1
                self.prompt_tokens += prompt if isinstance(prompt, int) else 0
                self.completion_tokens += completion if isinstance(completion, int) else 0

    def snapshot(self) -> Dict[str, int]:
        """Current totals as a plain dict."""
        with self._lock:
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "tokens_known": self.tokens_known,
            }

    def since(self, earlier: Dict[str, int]) -> Dict[str, int]:
        """Totals accrued since an earlier :meth:`snapshot`."""
        now = self.snapshot()
        return {key: now[key] - earlier.get(key, 0) for key in now}


class Message(TypedDict):
    """One chat message in the provider-neutral conversation format.

    Attributes:
        role: ``"system"``, ``"user"``, or ``"assistant"``.
        content: The message text.
    """

    role: str
    content: str

#: Provider prefixes recognized by :meth:`ModelWrapper.create` for explicit
#: routing (e.g. ``openai:anthropic.claude-sonnet-5`` forces the
#: OpenAI-compatible client for a Claude model id served by a gateway such as
#: the institutional gateway). ``paid-openai:`` bypasses an OpenAI-compatible gateway and
#: uses ``PAID_OPENAI_API_KEY`` against the public OpenAI endpoint. The prefix
#: is stripped before the id is sent to the API.
_PROVIDER_PREFIXES = ("openai", "paid-openai", "anthropic", "vllm")


def split_provider_prefix(model_name: str):
    r"""Split an optional ``provider:`` routing prefix off a model name.

    Only prefixes in :data:`_PROVIDER_PREFIXES` are treated as routing
    prefixes, so Windows paths (``C:\models\...``) and other colons in vLLM
    model paths pass through untouched.

    Args:
        model_name: A CLI model string, e.g. ``"openai:anthropic.claude-sonnet-5"``.

    Returns:
        ``(provider, bare_name)`` where ``provider`` is None when no
        recognized prefix is present.
    """
    prefix, sep, rest = model_name.partition(":")
    if sep and prefix.lower() in _PROVIDER_PREFIXES and rest:
        return prefix.lower(), rest
    return None, model_name


def short_model_name(model_name: str) -> str:
    """Filesystem-safe short name for a model: strips routing prefix and path.

    ``"openai:anthropic.claude-sonnet-5"`` -> ``"anthropic.claude-sonnet-5"``;
    ``"meta-llama/Llama-3-8b"`` -> ``"Llama-3-8b"``.
    """
    _, bare = split_provider_prefix(model_name)
    return bare.split("/")[-1]

class ModelWrapper(ABC):
    """Abstract base class for chat-model clients.

    Subclasses implement :meth:`generate`, :meth:`batch_generate`, and
    :meth:`batch_generate_with_probs` against a specific backend. Use
    :meth:`create` rather than instantiating a subclass directly so that
    provider routing (see the module docstring) is applied.

    The base class holds shared sampling parameters and the retry machinery:
    per-request exponential backoff (:meth:`_exponential_backoff`) plus a
    circuit breaker that raises after :attr:`max_consecutive_failures`
    exhausted-retry failures in a row (:meth:`_handle_api_failure`).

    Attributes:
        model_name: Model id sent to the backend (routing prefix stripped).
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens per request.
        top_p: Nucleus-sampling parameter (1.0 = default/off).
        max_retries: Attempts per request before giving up on it.
        initial_retry_delay: Backoff delay (seconds) after the first failure;
            doubles on each subsequent attempt.
        additional_params: Extra kwargs forwarded verbatim to the backend API.
        consecutive_failures: Count of back-to-back exhausted-retry failures.
        max_consecutive_failures: Circuit-breaker threshold for the above.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_consecutive_failures: int = 5,
        max_workers: int = 1,
        **kwargs
    ):
        """Store sampling parameters and initialize the failure tracker.

        Args:
            model_name: Model id understood by the concrete backend.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens per request.
            top_p: Nucleus-sampling parameter; 1.0 means "not set".
            max_retries: Attempts per request before returning ``''``.
            initial_retry_delay: Base delay (seconds) for exponential backoff.
            max_consecutive_failures: Exhausted-retry failures in a row after
                which the client raises instead of continuing.
            max_workers: Requests in flight per :meth:`batch_generate` call.
                The default of 1 keeps generation strictly sequential, which
                is what the upstream pipeline assumed. Raising it only changes
                scheduling -- the same prompts are sent and results stay in
                input order -- but a shared rate limit is spent faster, so
                raise it against a gateway you know tolerates the concurrency.
            **kwargs: Extra parameters forwarded to the backend on every call.
        """
        self.model_name = model_name
        self.requested_temperature = temperature
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        requested_retries = int(max_retries)
        if requested_retries < 1:
            logger.warning(
                "max_retries=%s is invalid; clamping to one attempt", max_retries
            )
        self.max_retries = max(1, requested_retries)
        self.initial_retry_delay = initial_retry_delay
        self.additional_params = kwargs
        
        self.max_workers = max(1, int(max_workers))

        #: Per-client tally of billed requests and tokens (see :class:`UsageLedger`).
        self.usage = UsageLedger()

        self.consecutive_failures = 0
        self.max_consecutive_failures = max_consecutive_failures

    @property
    def effective_temperature(self) -> Optional[float]:
        """Temperature currently sent to the provider, when one is sent."""
        return self.temperature

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate a completion for one conversation.

        Args:
            messages: The conversation so far.

        Returns:
            The model's response text. Implementations return ``''`` after
            ``max_retries`` failed attempts (having recorded the failure).
        """
        pass

    def _dispatch_batch(
        self, messages_list: List[List["Message"]], verbose: bool = False
    ) -> List[str]:
        """Run :meth:`generate` over many conversations, in input order.

        With ``max_workers == 1`` this is a plain sequential loop, identical to
        what the pipelines did before concurrency existed. Above 1, requests
        are issued through a thread pool -- the SDKs release the GIL while
        waiting on the network, so this converts a long serial wait into
        roughly ``max_workers``-way overlap without changing what is sent.

        Ordering is preserved by writing each result into its own slot rather
        than appending as futures complete, so callers that zip results back
        against their prompts stay correct.

        Args:
            messages_list: One conversation per prompt.
            verbose: Show a tqdm progress bar.

        Returns:
            One response string per conversation, in input order.
        """
        if self.max_workers <= 1 or len(messages_list) <= 1:
            iterator = tqdm(messages_list, desc='Batch Generation') if verbose else messages_list
            return [self.generate(messages) for messages in iterator]

        results: List[str] = [''] * len(messages_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.generate, messages): i
                for i, messages in enumerate(messages_list)
            }
            completed = as_completed(futures)
            if verbose:
                completed = tqdm(completed, total=len(futures), desc='Batch Generation')
            for future in completed:
                results[futures[future]] = future.result()
        return results

    @abstractmethod
    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        """Generate completions for several conversations.

        Args:
            messages_list: One conversation per prompt.

        Returns:
            One response string per conversation, in order.
        """
        pass

    @abstractmethod
    def batch_generate_with_probs(self, messages_list: List[List[Message]], outputs: List[str]) -> List[Dict[str, float]]:
        """Estimate a probability over candidate answers for each prompt.

        Intended for multiple-choice questions where each candidate in
        ``outputs`` is distinguished by the first response token (e.g.
        ``["A", "B"]``).

        Args:
            messages_list: One conversation per prompt.
            outputs: Candidate answer strings to score.

        Returns:
            For each prompt, a dict mapping every candidate to a probability
            (normalized to sum to 1; uniform when nothing can be measured).
        """
        pass

    @classmethod
    def create(cls, model_name: str, **kwargs) -> 'ModelWrapper':
        """Instantiate the right client for a model name.

        An explicit ``provider:`` prefix (see :data:`_PROVIDER_PREFIXES`)
        forces the client and is stripped from the id sent to the API — use
        ``openai:<model-id>`` for OpenAI-compatible gateways (e.g. an
        institutional gateway, LiteLLM, Open WebUI) that
        serve non-OpenAI models such as ``anthropic.claude-*`` ids, which
        would otherwise be routed by substring to the Anthropic SDK.
        Without a prefix, the provider is inferred: names containing ``gpt``
        go to OpenAI, names containing ``claude`` go to Anthropic, and
        anything else is loaded with vLLM.

        Throughput knobs: unless passed explicitly, ``max_workers`` comes
        from ``CONFLICTXGAME_MAX_WORKERS`` (default ``1``, strictly
        sequential) and ``max_retries`` from ``CONFLICTXGAME_MAX_RETRIES``
        (default ``3``). These are environment-driven rather than CLI flags
        because they tune throughput against a particular gateway rather
        than describing the experiment -- the same run is valid at any
        value. Lowering retries is what you want when the expected failure
        is a spent budget, which no amount of retrying will clear.

        Args:
            model_name: Model name, optionally with a ``provider:`` prefix.
            **kwargs: Forwarded to the client constructor (temperature,
                max_tokens, retry settings, extra API parameters).

        Returns:
            An :class:`OpenAIClient`, :class:`AnthropicClient`, or
            :class:`VLLMClient` instance.
        """
        kwargs.setdefault(
            "max_workers", _env_int("CONFLICTXGAME_MAX_WORKERS", 1)
        )
        kwargs.setdefault(
            "max_retries", _env_int("CONFLICTXGAME_MAX_RETRIES", 3)
        )
        provider, bare_name = split_provider_prefix(model_name)
        if provider == "openai":
            return OpenAIClient(bare_name, **kwargs)
        if provider == "paid-openai":
            paid_key = os.environ.get("PAID_OPENAI_API_KEY")
            if not paid_key:
                raise ValueError(
                    "paid-openai routing requires PAID_OPENAI_API_KEY"
                )
            return OpenAIClient(
                bare_name,
                api_key=paid_key,
                base_url=os.environ.get(
                    "PAID_OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                **kwargs,
            )
        if provider == "anthropic":
            return AnthropicClient(bare_name, **kwargs)
        if provider == "vllm":
            return VLLMClient(bare_name, **kwargs)
        if "gpt" in model_name.lower():
            return OpenAIClient(model_name, **kwargs)
        elif "claude" in model_name.lower():
            return AnthropicClient(model_name, **kwargs)
        else:
            return VLLMClient(model_name, **kwargs)

    def _exponential_backoff(self, attempt: int) -> None:
        """Sleep for ``initial_retry_delay * 2**attempt`` seconds between retries."""
        if attempt < self.max_retries:
            delay = self.initial_retry_delay * (2 ** attempt)
            time.sleep(delay)

    def _handle_api_failure(self, error_msg: str, context: str = ""):
        """Record an exhausted-retry failure; raise once the circuit-breaker trips."""
        self.consecutive_failures += 1
        logger.error(f"API failure #{self.consecutive_failures}: {error_msg}")
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            raise Exception(
                f"Model {self.model_name} failed {self.consecutive_failures} consecutive times. "
                f"Last error: {error_msg}. Context: {context}. Stopping execution."
            )
    
    def _handle_api_success(self, response=None):
        """Reset the failure counter and record the request in the ledger.

        Called at the billed point of every successful request, so it is also
        the natural place to count spend.

        Args:
            response: The raw provider response, when the caller has it. Its
                ``usage`` block (if any) supplies token counts; anything the
                ledger cannot read is still counted as one request.
        """
        self.consecutive_failures = 0
        self.usage.record(response)

class OpenAIClient(ModelWrapper):
    """Client for the OpenAI Chat Completions API and compatible gateways.

    The underlying ``OpenAI()`` SDK client is constructed with no arguments,
    so credentials and endpoint come from the environment
    (``OPENAI_API_KEY``, and ``OPENAI_BASE_URL`` for OpenAI-compatible
    gateways serving arbitrary models under this API — see the module
    docstring). Handles two model quirks adaptively: dropping ``temperature``
    permanently the first time a model rejects it, and retrying on empty
    responses from reasoning models (:class:`EmptyResponseError`).
    """

    def __init__(
        self, model_name: str, api_key: str | None = None,
        base_url: str | None = None, **kwargs,
    ):
        """Build the SDK client and apply gpt-5-family parameter adjustments.

        For model names containing ``gpt-5``, ``temperature`` is pinned to
        1.0 (non-default values are rejected) and ``max_tokens`` is raised to
        at least 1024 so reasoning tokens don't consume the whole budget.

        Args:
            model_name: Model id to request from the API.
            api_key: Optional client-specific key. When omitted, the OpenAI
                SDK reads ``OPENAI_API_KEY`` from the environment.
            base_url: Optional client-specific endpoint. Supplying this keeps
                a paid client independent of a gateway configured globally.
            **kwargs: Forwarded to :meth:`ModelWrapper.__init__`.

        Raises:
            ImportError: If the ``openai`` package is not installed.
        """
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for OpenAI models. "
                "Install it with: pip install openai"
            )
        super().__init__(model_name, **kwargs)
        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        # Some models reject `temperature` outright (rather than only rejecting
        # non-default values). We can't know which from the id alone — gateways
        # front many providers — so this starts True and is switched off
        # permanently the first time the API says the parameter is unsupported.
        self._send_temperature = True
        if 'gpt-5' in model_name.lower():
            # gpt-5-family reasoning models reject non-default temperature and
            # need completion-token headroom for reasoning; treat 1024 as a
            # floor rather than overriding an explicitly larger request.
            if self.requested_temperature != 1.0:
                logger.warning(
                    "%s requested temperature %s, but the gpt-5 provider "
                    "constraint pins the effective temperature to 1.0",
                    model_name, self.requested_temperature,
                )
            self.temperature = 1.0
            self.max_tokens = max(self.max_tokens, 1024)

    @property
    def effective_temperature(self) -> Optional[float]:
        """The value sent to OpenAI, or ``None`` after parameter fallback."""
        return self.temperature if self._send_temperature else None

    @staticmethod
    def _is_unsupported_temperature_error(error: Exception) -> bool:
        """Whether an API error says this model does not accept `temperature`.

        Gateways surface the upstream provider's message, e.g. Vertex-served
        Claude via LiteLLM: ``` `temperature` is deprecated for this model. ```
        """
        message = str(error).lower()
        return 'temperature' in message and (
            'deprecated' in message or 'unsupported' in message or 'not supported' in message
        )

    def _request_kwargs(self, messages: List[Message]) -> Dict[str, Any]:
        """Assemble request kwargs, omitting parameters this model rejects."""
        request_kwargs = dict(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.max_tokens,
            **self.additional_params
        )
        if self._send_temperature:
            request_kwargs['temperature'] = self.temperature
        # top_p=1.0 is a no-op and newer models (gpt-5 family,
        # claude-sonnet-5 via gateways) reject the parameter outright,
        # so it is only sent when explicitly set away from the default.
        if self.top_p != 1.0:
            request_kwargs['top_p'] = self.top_p
        return request_kwargs

    def generate(self, messages: List[Message]) -> str:
        """Generate one completion with retries.

        Empty/null content (a reasoning model spending its budget on
        reasoning tokens) and malformed responses (see
        :func:`completion_text`) are both treated as retryable. If the API
        says the model rejects ``temperature``, the parameter is dropped for
        the client's lifetime and the request is reissued without backoff.

        Args:
            messages: The conversation so far.

        Returns:
            The response text, or ``''`` after ``max_retries`` failures
            (recorded via the circuit breaker).
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**self._request_kwargs(messages))
                self._handle_api_success(response)
                # Before completion_text, which raises on empty content: a
                # reasoning model that spent its whole budget thinking is
                # exactly the call worth keeping a trace of. No-op unless
                # CONFLICTXGAME_CAPTURE_REASONING names a file.
                capture_response(self.model_name, messages, response)
                return completion_text(response)
            # A provider that returns a malformed body surfaces as JSONDecodeError
            # from the SDK, not APIError. Retry it like any transient failure: left
            # uncaught it kills the whole run before any row is written.
            except (APIError, EmptyResponseError, json.JSONDecodeError) as e:
                if self._send_temperature and self._is_unsupported_temperature_error(e):
                    # Drop the parameter for the rest of this client's life and
                    # retry without backoff. This happens at most once, so it
                    # cannot loop even though it consumes one attempt.
                    logger.info(
                        f"{self.model_name} rejects `temperature`; retrying without it."
                    )
                    self._send_temperature = False
                    if attempt == self.max_retries - 1:
                        self._handle_api_failure(str(e), messages)
                        return ''
                    continue
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    self._handle_api_failure(str(e), messages)
                    return ''
                self._exponential_backoff(attempt)
        return ''


    def batch_generate(self, messages_list: List[List[Message]], verbose: bool = False) -> List[str]:
        """Generate completions sequentially (one API call per prompt).

        Args:
            messages_list: One conversation per prompt.
            verbose: Show a tqdm progress bar.

        Returns:
            One response string per conversation, in order.
        """
        return self._dispatch_batch(messages_list, verbose=verbose)
        
    def batch_generate_with_probs(self, messages_list: List[List[Message]], outputs: List[str]) -> List[Dict[str, float]]:
        """Score MCQ candidates from first-token logprobs.

        Temporarily switches to ``max_tokens=5``, ``temperature=0.0``, and
        requests ``top_logprobs`` for the first completion token; each
        candidate's probability is the (renormalized) exponentiated logprob
        mass of first tokens matching it case-insensitively. Original
        sampling settings are restored afterwards.

        Args:
            messages_list: One conversation per prompt.
            outputs: Candidate answers distinguished by their first token.

        Returns:
            Per prompt, ``{candidate: probability}`` summing to 1; uniform
            when no candidate token appears in the logprobs or the API errors.
        """

        original_max_tokens = self.max_tokens
        original_temperature = self.temperature
        original_additional_params = self.additional_params.copy()
        
        if original_max_tokens > 10:
            logger.warning(f"batch_generate_with_probs called with high max_tokens={original_max_tokens}. "
                          f"Setting to 1 to save tokens since only the first token is needed.")
        
        try:
            self.max_tokens = 5  # Only need the first token
            self.temperature = 0.0  # Need deterministic outputs for logprobs
            
            self.additional_params = original_additional_params.copy()
            self.additional_params["logprobs"] = True
            self.additional_params["top_logprobs"] = len(outputs) + 5  # Request more than needed
            
            results = []
            for messages in messages_list:
                try:
                    try:
                        response = self.client.chat.completions.create(
                            **self._request_kwargs(messages)
                        )
                    except APIError as e:
                        if not (self._send_temperature
                                and self._is_unsupported_temperature_error(e)):
                            raise
                        # See generate(): this model rejects `temperature`, so
                        # drop it permanently and reissue. Determinism is lost,
                        # but the alternative is no logprobs at all.
                        logger.info(
                            f"{self.model_name} rejects `temperature`; retrying without it."
                        )
                        self._send_temperature = False
                        response = self.client.chat.completions.create(
                            **self._request_kwargs(messages)
                        )

                    first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                    
                    probs = {}
                    logprob_sum = 0
                    for output in outputs:
                        probs[output] = 0
                        for token_logprob in first_token_logprobs:
                            if token_logprob.token.strip().upper() == output.strip().upper():
                                probs[output] += np.exp(token_logprob.logprob)
                        logprob_sum += probs[output]
                    
                    if logprob_sum > 0:
                        for output in probs:
                            probs[output] /= logprob_sum
                    else:
                        logger.warning("No valid logprobs found for any of the target outputs. Using uniform probabilities.")
                        for output in outputs:
                            probs[output] = 1.0/len(outputs)
                    
                    results.append(probs)
                    
                except (APIError, AttributeError, IndexError, KeyError, TypeError) as e:
                    # Non-APIError shapes cover the gateway's bare-string
                    # responses (see completion_text): fall back to uniform
                    # probabilities rather than aborting the batch.
                    logger.error(f"OpenAI API error when getting logprobs: {str(e)}")
                    results.append({output: 1.0/len(outputs) for output in outputs})
                    
            return results
            
        finally:
            self.max_tokens = original_max_tokens
            self.temperature = original_temperature
            self.additional_params = original_additional_params

class AnthropicClient(ModelWrapper):
    """Client for Claude models via the Anthropic Messages API.

    Credentials come from the environment (``ANTHROPIC_API_KEY``). The
    Messages API takes the system prompt as a separate parameter, so
    :meth:`generate` splits a leading ``system`` message out of the
    conversation. Because the API does not expose logprobs,
    :meth:`batch_generate_with_probs` estimates MCQ probabilities by
    repeated sampling instead.
    """

    _BATCH_MANIFEST_VERSION = 1

    def __init__(
        self,
        model_name: str,
        *,
        use_message_batches: bool = False,
        batch_state_dir: Optional[str] = None,
        batch_poll_interval: float = 60.0,
        **kwargs,
    ):
        """Build the Anthropic SDK client.

        Args:
            model_name: Claude model id to request from the API.
            use_message_batches: Route :meth:`batch_generate` through
                Anthropic's asynchronous Message Batches API.  This is an
                explicit opt-in; the default retains the original synchronous
                per-prompt behavior.
            batch_state_dir: Directory for crash-resumable batch manifests.
                Required when ``use_message_batches`` is true.
            batch_poll_interval: Seconds between batch status polls.  Zero is
                useful in tests; production callers should normally use 60.
            **kwargs: Forwarded to :meth:`ModelWrapper.__init__`.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
            ValueError: If Message Batches are enabled without a state
                directory, or the poll interval is negative.
        """
        if Anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for Claude models. "
                "Install it with: pip install anthropic"
            )
        if use_message_batches and not batch_state_dir:
            raise ValueError(
                "batch_state_dir is required when use_message_batches=True"
            )
        if batch_poll_interval < 0:
            raise ValueError("batch_poll_interval must be non-negative")
        super().__init__(model_name, **kwargs)
        self.use_message_batches = bool(use_message_batches)
        self.batch_state_dir = (
            os.path.abspath(os.fspath(batch_state_dir))
            if batch_state_dir is not None else None
        )
        self.batch_poll_interval = float(batch_poll_interval)
        # A completed manifest may be requested twice by orchestration code.
        # Its spend belongs in a fresh process's ledger, but never twice in the
        # same process.
        self._accounted_batch_results = set()
        self._accounted_batch_failures = set()
        self.client = Anthropic()

    def _anthropic_request_params(self, messages: List[Message]) -> dict:
        """Build parameters shared verbatim by sync and batch requests."""
        if messages and messages[0]['role'] == 'system':
            system = messages[0]['content']
            provider_messages = messages[1:]
        else:
            system = 'You are a helpful assistant.'
            provider_messages = messages

        request_kwargs = dict(
            model=self.model_name,
            system=system,
            messages=provider_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.additional_params,
        )
        if self.top_p != 1.0:  # newer Claude models reject default top_p
            request_kwargs['top_p'] = self.top_p
        return request_kwargs

    @staticmethod
    def _anthropic_message_text(response) -> str:
        """Return the first text block from an Anthropic Message response."""
        if isinstance(response, str):
            raise EmptyResponseError(
                f"API returned a bare string instead of a message: {response[:200]!r}"
            )
        try:
            blocks = list(response.content)
        except (AttributeError, TypeError) as e:
            raise EmptyResponseError(
                f"Malformed message response ({type(response).__name__}): {e}"
            ) from e
        # content[0] is not the answer when thinking is on: the SDK then
        # returns [thinking, text].
        for block in blocks:
            if getattr(block, 'type', None) == 'text' and isinstance(
                getattr(block, 'text', None), str
            ):
                return block.text
        raise EmptyResponseError(
            "No text block in message response; got block types "
            f"{[getattr(block, 'type', type(block).__name__) for block in blocks]}"
        )

    @staticmethod
    def _batch_fingerprint(requests: List[dict]) -> str:
        """Fingerprint ordered provider parameters for safe manifest reuse."""
        try:
            encoded = json.dumps(
                requests, sort_keys=True, separators=(',', ':'), ensure_ascii=False
            ).encode('utf-8')
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Anthropic batch request parameters must be JSON-serializable "
                "so they can be resumed safely"
            ) from e
        return hashlib.sha256(encoded).hexdigest()

    def _batch_manifest_path(self, batch_key: str) -> str:
        if not isinstance(batch_key, str) or not batch_key.strip():
            raise ValueError(
                "batch_key is required when Anthropic Message Batches are enabled"
            )
        assert self.batch_state_dir is not None
        os.makedirs(self.batch_state_dir, exist_ok=True)
        key_digest = hashlib.sha256(batch_key.encode('utf-8')).hexdigest()
        return os.path.join(self.batch_state_dir, f'{key_digest}.json')

    @staticmethod
    def _write_batch_manifest(path: str, manifest: dict) -> None:
        """Atomically replace one UTF-8 JSON batch manifest."""
        tmp_path = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            replace_with_retry(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                logger.warning("Could not remove temporary batch manifest %s", tmp_path)

    @staticmethod
    def _load_batch_manifest(path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as e:
            raise MessageBatchStateError(
                f"Cannot read Anthropic batch manifest {path}: {e}"
            ) from e
        if not isinstance(manifest, dict):
            raise MessageBatchStateError(
                f"Anthropic batch manifest {path} is not a JSON object"
            )
        return manifest

    def _batch_api_call(self, operation: str, call):
        """Retry one Message Batches control-plane operation."""
        for attempt in range(self.max_retries):
            try:
                return call()
            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                logger.warning(
                    "Anthropic batch %s error (attempt %d/%d): %s",
                    operation, attempt + 1, self.max_retries, e,
                )
                if attempt == self.max_retries - 1:
                    self._handle_api_failure(str(e), operation)
                    raise
                self._exponential_backoff(attempt)
        raise AssertionError("max_retries must be positive")

    @staticmethod
    def _batch_usage(response) -> dict:
        usage = getattr(response, 'usage', None)
        if usage is None:
            return {}
        fields = (
            'input_tokens', 'output_tokens', 'prompt_tokens',
            'completion_tokens', 'cache_creation_input_tokens',
            'cache_read_input_tokens',
        )
        return {
            field: getattr(usage, field)
            for field in fields
            if isinstance(getattr(usage, field, None), int)
        }

    @staticmethod
    def _batch_error_details(batch_result) -> dict:
        outer = getattr(batch_result, 'error', None)
        inner = getattr(outer, 'error', outer)
        return {
            'type': getattr(inner, 'type', None),
            'message': getattr(inner, 'message', str(inner) if inner else None),
        }

    def _account_batch_records(self, manifest_path: str, records: List[dict]) -> None:
        """Fold persisted individual results into this process's ledger once."""
        successes = 0
        for record in records:
            if record.get('type') != 'succeeded':
                continue
            successes += 1
            identity = (manifest_path, record.get('custom_id'))
            if identity in self._accounted_batch_results:
                continue
            usage = SimpleNamespace(**record.get('usage', {}))
            self._handle_api_success(SimpleNamespace(usage=usage))
            self._accounted_batch_results.add(identity)

        failure_identity = (manifest_path, 'all_failed')
        if records and not successes and failure_identity not in self._accounted_batch_failures:
            self._handle_api_failure(
                "Anthropic Message Batch ended without a successful request",
                manifest_path,
            )
            self._accounted_batch_failures.add(failure_identity)

    def _message_batch_generate(
        self,
        messages_list: List[List[Message]],
        batch_key: str,
    ) -> List[str]:
        """Submit or resume one persistent Anthropic Message Batch."""
        if not messages_list:
            return []

        request_params = [
            self._anthropic_request_params(messages) for messages in messages_list
        ]
        requests = [
            {'custom_id': f'request_{index:06d}', 'params': params}
            for index, params in enumerate(request_params)
        ]
        fingerprint = self._batch_fingerprint(requests)
        manifest_path = self._batch_manifest_path(batch_key)

        if os.path.exists(manifest_path):
            manifest = self._load_batch_manifest(manifest_path)
            expected = {
                'version': self._BATCH_MANIFEST_VERSION,
                'batch_key': batch_key,
                'fingerprint': fingerprint,
                'request_count': len(requests),
            }
            mismatches = {
                key: (manifest.get(key), value)
                for key, value in expected.items()
                if manifest.get(key) != value
            }
            if mismatches:
                raise MessageBatchStateError(
                    f"Anthropic batch_key {batch_key!r} was reused with different "
                    f"state or requests: {mismatches}"
                )
            batch_id = manifest.get('batch_id')
            if not isinstance(batch_id, str) or not batch_id:
                raise MessageBatchStateError(
                    f"Anthropic batch manifest {manifest_path} has no batch_id"
                )
            records = manifest.get('results')
            if manifest.get('status') == 'ended':
                if not isinstance(records, list):
                    raise MessageBatchStateError(
                        f"Anthropic batch manifest {manifest_path} has invalid results"
                    )
                ordered = _records_in_request_order(records, requests, batch_id)
                self._account_batch_records(manifest_path, ordered)
                return [record['text'] for record in ordered]
        else:
            created = self._batch_api_call(
                'create', lambda: self.client.messages.batches.create(requests=requests)
            )
            batch_id = getattr(created, 'id', None)
            if not isinstance(batch_id, str) or not batch_id:
                raise MessageBatchStateError(
                    "Anthropic Message Batches create returned no batch id"
                )
            manifest = {
                'version': self._BATCH_MANIFEST_VERSION,
                'batch_key': batch_key,
                'fingerprint': fingerprint,
                'request_count': len(requests),
                'batch_id': batch_id,
                'status': getattr(created, 'processing_status', 'in_progress'),
                'results': None,
            }
            self._write_batch_manifest(manifest_path, manifest)

        while manifest.get('status') != 'ended':
            batch = self._batch_api_call(
                'retrieve',
                lambda: self.client.messages.batches.retrieve(batch_id),
            )
            status = getattr(batch, 'processing_status', None)
            if status not in ('in_progress', 'canceling', 'ended'):
                raise MessageBatchStateError(
                    f"Anthropic batch {batch_id} returned unknown status {status!r}"
                )
            manifest['status'] = status
            self._write_batch_manifest(manifest_path, manifest)
            if status != 'ended' and self.batch_poll_interval:
                logger.info(
                    "Anthropic batch %s is %s; polling again in %.1f seconds",
                    batch_id, status, self.batch_poll_interval,
                )
                time.sleep(self.batch_poll_interval)

        individual_results = self._batch_api_call(
            'results', lambda: list(self.client.messages.batches.results(batch_id))
        )
        records = []
        for item in individual_results:
            custom_id = getattr(item, 'custom_id', None)
            result = getattr(item, 'result', None)
            result_type = getattr(result, 'type', None)
            if result_type == 'succeeded':
                message = getattr(result, 'message', None)
                try:
                    response_text = self._anthropic_message_text(message)
                except EmptyResponseError as e:
                    logger.warning(
                        "Anthropic batch request %s had no usable text: %s",
                        custom_id, e,
                    )
                    response_text = ''
                record = {
                    'custom_id': custom_id,
                    'type': 'succeeded',
                    'text': response_text,
                    'usage': self._batch_usage(message),
                }
            elif result_type in ('errored', 'canceled', 'expired'):
                record = {
                    'custom_id': custom_id,
                    'type': result_type,
                    'text': '',
                    'usage': {},
                }
                if result_type == 'errored':
                    record['error'] = self._batch_error_details(result)
                logger.warning(
                    "Anthropic batch request %s ended as %s%s",
                    custom_id,
                    result_type,
                    f": {record.get('error')}" if record.get('error') else '',
                )
            else:
                raise MessageBatchStateError(
                    f"Anthropic batch {batch_id} returned unknown result type "
                    f"{result_type!r} for {custom_id}"
                )
            records.append(record)

        records = _records_in_request_order(records, requests, batch_id)
        manifest['status'] = 'ended'
        manifest['results'] = records
        self._write_batch_manifest(manifest_path, manifest)
        self._account_batch_records(manifest_path, records)
        return [record['text'] for record in records]
    
    def generate(self, messages: List[Message]) -> str:
        """Generate one completion with retries.

        A leading ``system`` message is passed via the API's ``system``
        parameter (defaulting to ``'You are a helpful assistant.'`` when
        absent). ``top_p`` is only sent when set away from 1.0, since newer
        Claude models reject the default value. The reply is the first
        ``text`` content block: models with thinking on by default (Sonnet 5
        and later) put a ``thinking`` block ahead of it, and a reply with no
        text block at all is treated as retryable.

        Args:
            messages: The conversation so far.

        Returns:
            The response text, or ``''`` after ``max_retries`` failures
            (recorded via the circuit breaker).
        """

        request_kwargs = self._anthropic_request_params(messages)
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**request_kwargs)
                self._handle_api_success(response)
                return self._anthropic_message_text(response)

            except (APIConnectionError, RateLimitError, APIStatusError, EmptyResponseError) as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed after {self.max_retries} attempts: {str(e)} "
                        f"for prompt {request_kwargs.get('messages', messages)}"
                    )
                    self._handle_api_failure(
                        str(e), request_kwargs.get('messages', messages)
                    )
                    return ''
                self._exponential_backoff(attempt)
        return ''

    def batch_generate(
        self,
        messages_list: List[List[Message]],
        verbose: bool = False,
        batch_key: Optional[str] = None,
    ) -> List[str]:
        """Generate completions synchronously or through Message Batches.

        Args:
            messages_list: One conversation per prompt.
            verbose: Show a tqdm progress bar.
            batch_key: Stable identity for one logical asynchronous batch.
                Required only when ``use_message_batches`` is enabled. Reusing
                it with identical inputs resumes the persisted batch; reusing
                it with different inputs raises :class:`MessageBatchStateError`.

        Returns:
            One response string per conversation, in order.
        """
        if self.use_message_batches:
            if batch_key is None:
                raise ValueError(
                    "batch_key is required when Anthropic Message Batches are enabled"
                )
            return self._message_batch_generate(messages_list, batch_key)
        return self._dispatch_batch(messages_list, verbose=verbose)
        
    def batch_generate_with_probs(self, messages_list: List[List[Message]], outputs: List[str], num_samples: int = 3) -> List[Dict[str, float]]:
        """Estimate MCQ probabilities by repeated sampling (no logprobs API).

        Each prompt is sampled ``num_samples`` times at ``temperature=1.0``
        with ``max_tokens=5``; a candidate's probability is the fraction of
        samples whose first response character matches it (case-insensitive).
        Original sampling settings are restored afterwards.

        Args:
            messages_list: One conversation per prompt.
            outputs: Candidate answers distinguished by their first character.
            num_samples: Samples drawn per prompt.

        Returns:
            Per prompt, ``{candidate: probability}`` summing to 1; uniform
            when no sample matches any candidate.
        """
        results = []
        
        original_max_tokens = self.max_tokens
        original_temperature = self.temperature
        
        try:
            self.max_tokens = 5  # Only need the first token
            self.temperature = 1.0  # High temperature for diverse sampling
            
            for messages in messages_list:
                batch_messages = [messages] * num_samples
                
                responses = self.batch_generate(batch_messages)
                
                processed_responses = []
                for response in responses:
                    first_token = response.strip().upper()
                    if first_token and first_token[0] in [o.strip().upper() for o in outputs]:
                        processed_responses.append(first_token[0])
                
                counter = Counter(processed_responses)
                total = len(processed_responses)
                
                if total > 0:
                    probs = {output: counter[output.strip().upper()] / total for output in outputs}
                else:
                    logger.warning("No valid samples found for any of the target outputs. Using uniform probabilities.")
                    probs = {output: 1.0/len(outputs) for output in outputs}
                    
                results.append(probs)
                
        finally:
            self.max_tokens = original_max_tokens
            self.temperature = original_temperature
        
        return results

#: Process-wide cache of loaded vLLM engines, keyed by model name, so that
#: multiple :class:`VLLMClient` instances for the same model share one copy
#: of the weights on the GPUs.
_vllm_instances = {}

class VLLMClient(ModelWrapper):
    """Client for open-weight models hosted in-process with vLLM.

    Engines are cached in :data:`_vllm_instances` and shared across client
    instances. vLLM is used here without a chat template, so conversations
    are rendered to a single prompt string by :meth:`format_messages`, which
    dispatches on the model name to a family-specific formatter (Tulu, OLMo,
    WildGuard, Llama, Qwen, Gemma, Mistral).
    """

    def __init__(self, model_name: str, **kwargs):
        """Load (or reuse) the vLLM engine and build sampling parameters.

        The engine is created with ``gpu_memory_utilization=0.9``,
        ``max_model_len=4096``, and ``tensor_parallel_size`` from
        :func:`gpus_needed`. Mistral models additionally disable custom
        all-reduce and install ``MistralTokenizer``.

        Args:
            model_name: Hugging Face id or local path of the model.
            **kwargs: Forwarded to :meth:`ModelWrapper.__init__`; the extra
                parameters become ``SamplingParams`` fields.

        Raises:
            ImportError: If the ``vllm`` package is not installed.
            Exception: If the engine fails to initialize.
        """
        if LLM is None:
            raise ImportError(
                "The 'vllm' package is required for open-weight models. "
                "Install it with: pip install vllm"
            )
        super().__init__(model_name, **kwargs)
        global _vllm_instances
        try:
            if model_name not in _vllm_instances:
                logger.info(f"Loading VLLM Model {model_name}")
                llm_kwargs = {
                    'model': model_name,
                    'gpu_memory_utilization': 0.9,
                    'tensor_parallel_size': gpus_needed(model_name),
                    'max_model_len': 4096
                }
                if 'mistral' in model_name.lower():
                    llm_kwargs['disable_custom_all_reduce'] = True
                _vllm_instances[model_name] = LLM(**llm_kwargs)

            self.llm = _vllm_instances[model_name]
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                **self.additional_params
            )
            if 'mistral' in model_name.lower():
                from vllm.transformers_utils.tokenizers import MistralTokenizer
                tokenizer = MistralTokenizer.from_pretrained(model_name)
                self.llm.set_tokenizer(tokenizer)

        except Exception as e:
            raise Exception(f"Failed to initialize vLLM model: {str(e)}")
    
    def format_messages(self, messages: List[Message]) -> str:
        """Render a conversation to a prompt string for this model family.

        Dispatches on substrings of the model name (checked in order: tulu,
        olmo, wildguard, llama, qwen, gemma, mistral) to the matching
        ``format_messages_for_*`` helper.

        Args:
            messages: The conversation to render.

        Returns:
            The formatted prompt string.

        Raises:
            NotImplementedError: If no formatter matches the model name.
        """
        messages = messages.copy()
        if 'tulu' in self.model_name.lower():
            return(self.format_messages_for_tulu(messages))
        elif 'olmo' in self.model_name.lower():
            return(self.format_messages_for_olmo(messages))
        elif 'wildguard' in self.model_name.lower():
            return(self.format_messages_for_wildguard(messages))
        elif 'llama' in self.model_name.lower():
            return(self.format_messages_for_llama(messages))
        elif 'qwen' in self.model_name.lower():
            return(self.format_messages_for_qwen(messages))
        elif 'gemma' in self.model_name.lower():
            return(self.format_messages_for_gemma(messages))
        elif 'mistral' in self.model_name.lower():
            return(self.format_messages_for_mistral(messages))
        else:
            raise NotImplementedError(f"Message formatting not implemented for model {self.model_name}")
        
    def format_messages_for_tulu(self, messages: List[Message]) -> str:
        """Render messages in the Tulu chat format.

        Uses ``<|user|>``/``<|assistant|>`` tags with ``<|endoftext|>``
        after assistant turns. A system message is prepended to the first
        user message (Tulu has no system tag), and a trailing
        ``<|assistant|>`` tag is appended when the last turn is a user turn.
        """
        formatted_str = ""
        for message in messages:
            if message['role'] == 'user':
                formatted_str += f"<|user|>\n{message['content']}\n"
            elif message['role'] == 'assistant':
                formatted_str += f"<|assistant|>\n{message['content']}<|endoftext|>\n"
            elif message['role'] == 'system':
                # For system messages, prepend to the first user message
                for i, next_msg in enumerate(messages):
                    if next_msg['role'] == 'user':
                        next_msg['content'] = message['content'] + '\n\n' + next_msg['content']
                        break
        
        # If the last message was from a user, add the assistant tag
        if messages[-1]['role'] == 'user':
            formatted_str += "<|assistant|>\n"
            
        return formatted_str

    def format_messages_for_olmo(self, messages: List[Message]) -> str:
        """Render messages in the OLMo chat format.

        Identical to the Tulu format except that the prompt begins with an
        ``<|endoftext|>`` token.
        """
        formatted_str = "<|endoftext|>"  # Start with endoftext token
        for message in messages:
            if message['role'] == 'user':
                formatted_str += f"<|user|>\n{message['content']}\n"
            elif message['role'] == 'assistant':
                formatted_str += f"<|assistant|>\n{message['content']}<|endoftext|>\n"
            elif message['role'] == 'system':
                # For system messages, prepend to the first user message
                for i, next_msg in enumerate(messages):
                    if next_msg['role'] == 'user':
                        next_msg['content'] = message['content'] + '\n\n' + next_msg['content']
                        break
        
        # If the last message was from a user, add the assistant tag
        if messages[-1]['role'] == 'user':
            formatted_str += "<|assistant|>\n"
            
        return formatted_str
    
    def format_messages_for_wildguard(self, messages: List[Message]) -> str:
        """Render a (prompt, response) pair in WildGuard's classifier format.

        WildGuard is a safety classifier, not a chat model: the two messages
        are inserted into its fixed harmfulness/refusal instruction template.

        Raises:
            ValueError: If ``messages`` is not exactly a user/assistant pair.
        """
        if len(messages) != 2:
            raise ValueError("WildGuard model only supports 2 messages: user and assistant")
        prompt = messages[0]['content']
        response = messages[1]['content']
        instruction_format = """<s><|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:
{response}

---

Answers: [/INST]
<|assistant|>
"""
        return instruction_format.format(prompt=prompt, response=response)

    def format_messages_for_llama(self, messages: List[Message]) -> str:
        """Render messages in the Llama 3 chat format.

        Each message becomes a ``<|start_header_id|>role<|end_header_id|>``
        block ending in ``<|eot_id|>``, prefixed by ``<|begin_of_text|>``; a
        header for the next speaker (assistant, or user if the last turn was
        an assistant turn) is appended to cue the generation.
        """
        formatted_msg = []
        for message in messages:
            formatted_msg.append(f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}<|eot_id|>")
        formatted_str = '<|begin_of_text|>' + '\n\n'.join(formatted_msg)

        if messages[-1]['role'] == 'assistant':
            formatted_str += f"<|start_header_id|>user<|end_header_id|>\n\n"
        else:
            formatted_str += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return(formatted_str)
    
    def format_messages_for_qwen(self, messages: List[Message]) -> str:
        """Render messages in the Qwen (ChatML) format.

        Each message becomes an ``<|im_start|>role ... <|im_end|>`` block; an
        opening tag for the next speaker is appended to cue the generation.
        """
        formatted_msg = []
        for message in messages:
            formatted_msg.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        formatted_str = '\n'.join(formatted_msg)
        
        if messages[-1]['role'] == 'assistant':
            formatted_str += f"\n<|im_start|>user\n"
        else:
            formatted_str += f"\n<|im_start|>assistant\n"
        return(formatted_str)

    def format_messages_for_gemma(self, messages: List[Message]) -> str:
        """Render messages in the Gemma chat format.

        Uses ``<start_of_turn>user`` / ``<start_of_turn>model`` blocks. A
        system message is prepended to the first user message (Gemma has no
        system role), and an opening tag for the next speaker is appended.
        """
        formatted_str = ''
        for message in messages:
            if message['role'] == 'system':
                for i, next_msg in enumerate(messages):
                    if next_msg['role'] == 'user':
                        next_msg['content'] = message['content'] + '\n\n' + next_msg['content']
                        break
            elif message['role'] == 'user':
                formatted_str += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
            elif message['role'] == 'assistant':
                formatted_str += f"<start_of_turn>model\n{message['content']}<end_of_turn>\n"
        
        if messages[-1]['role'] == 'assistant':
            formatted_str += f"<start_of_turn>user\n"
        else:
            formatted_str += f"<start_of_turn>model\n"
        return(formatted_str)

    def format_messages_for_mistral(self, messages: List[Message]) -> str:
        """Render messages in the Mistral instruction format.

        User turns are wrapped in ``[INST] ... [/INST]`` after a ``<s>``
        token; assistant turns end with ``</s>``. A system message is
        prepended to the first user message (Mistral has no system role).
        """
        bos_token = "<s>"
        eos_token = "</s>"
        
        formatted_str = bos_token
        
        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
                for i, next_msg in enumerate(messages):
                    if next_msg['role'] == 'user':
                        next_msg['content'] = system_message + '\n\n' + next_msg['content']
                        break
        
        for i, message in enumerate(messages):
            if message['role'] == 'user':
                formatted_str += f" [INST] {message['content']} [/INST]"
                if i < len(messages) - 1:
                    formatted_str += " "
            elif message['role'] == 'assistant':
                formatted_str += f"{message['content']}{eos_token}"
                if i < len(messages) - 1:
                    formatted_str += " "

        return formatted_str

    def generate(self, messages: List[Message]) -> str:
        """Generate one completion locally.

        Args:
            messages: The conversation, rendered via :meth:`format_messages`.

        Returns:
            The generated text.
        """
        formatted_msg = self.format_messages(messages)
        response = self.llm.generate([formatted_msg], sampling_params=self.sampling_params, use_tqdm=False)
        return response[0].outputs[0].text

    def batch_generate(self, messages_list: List[List[Message]], verbose: bool = False) -> List[str]:
        """Generate completions for all prompts in one true vLLM batch.

        Args:
            messages_list: One conversation per prompt.
            verbose: Show vLLM's tqdm progress bar.

        Returns:
            One response string per conversation, in order.
        """
        formatted_msgs = [self.format_messages(messages) for messages in messages_list]
        if verbose:
            response = self.llm.generate(formatted_msgs, sampling_params=self.sampling_params, use_tqdm=True)
        else:
            response = self.llm.generate(formatted_msgs, sampling_params=self.sampling_params, use_tqdm=False)
        return [out.outputs[0].text for out in response]
        
    def batch_generate_with_probs(self, messages_list: List[List[Message]], outputs: List[str]) -> List[Dict[str, float]]:
        """Score MCQ candidates from first-token logprobs in one vLLM batch.

        Temporarily swaps in deterministic ``SamplingParams`` (temperature 0,
        one token, ``logprobs=len(outputs) + 5``); each candidate's
        probability is the (renormalized) exponentiated logprob mass of
        first tokens whose decoded text matches it case-insensitively.
        Original sampling settings are restored afterwards.

        Args:
            messages_list: One conversation per prompt.
            outputs: Candidate answers distinguished by their first token.

        Returns:
            Per prompt, ``{candidate: probability}`` summing to 1; uniform
            when no candidate token appears or generation fails.
        """
        original_max_tokens = self.max_tokens
        original_temperature = self.temperature
        original_sampling_params = self.sampling_params
        
        if original_max_tokens > 10:
            logger.warning(f"batch_generate_with_probs called with high max_tokens={original_max_tokens}. "
                          f"Setting to 1 to save tokens since only the first token is needed.")
                          
        try:
            self.max_tokens = 5  # Only need the first token
            self.temperature = 0.0  # Need deterministic output
            
            # Create sampling parameters for logprobs
            self.sampling_params = SamplingParams(
                temperature=0,  # Use temperature 0 for deterministic output
                max_tokens=1,   # Only need the first token
                top_p=1.0,
                logprobs=len(outputs) + 5,  # Request more than needed to ensure we get all outputs
            )
            
            # Format messages
            formatted_msgs = [self.format_messages(messages) for messages in messages_list]
            
            # Get responses with logprobs
            results = []
            
            try:
                responses = self.llm.generate(formatted_msgs, sampling_params=self.sampling_params, use_tqdm=False)
                
                for response in responses:
                    token_logprobs = response.outputs[0].logprobs[0]

                    # Filter logprobs for our outputs of interest and convert to probabilities
                    probs = {}
                    logprob_sum = 0
                    for output in outputs:
                        probs[output] = 0
                        output_upper = output.strip().upper()
                        
                        # Check each token's decoded representation
                        for token_id, logprob_obj in token_logprobs.items():
                            if logprob_obj.decoded_token.strip().upper() == output_upper:
                                probs[output] += np.exp(logprob_obj.logprob)
                        
                        logprob_sum += probs[output]
                    
                    if logprob_sum > 0:
                        for output in probs:
                            probs[output] /= logprob_sum
                    else:
                        logger.warning("No valid logprobs found for any of the target outputs. Using uniform probabilities.")
                        for output in outputs:
                            probs[output] = 1.0 / len(outputs)
                    
                    results.append(probs)
                    
            except Exception as e:
                logger.error(f"vLLM error when getting logprobs: {str(e)}")
                # On error, return equal probabilities for all messages
                for _ in range(len(messages_list)):
                    results.append({output: 1.0/len(outputs) for output in outputs})
            
            return results
                
        finally:
            self.max_tokens = original_max_tokens
            self.temperature = original_temperature
            self.sampling_params = original_sampling_params
