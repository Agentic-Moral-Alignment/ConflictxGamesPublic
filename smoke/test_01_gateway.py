"""Gateway connectivity and routing — the cheapest checks, run first.

If these fail, nothing else in the suite can work: the key, the base URL, or
the model ids are wrong.
"""

import pytest

from conftest import (
    GEN_MODEL,
    JUDGE_MODEL,
    TARGET_MODEL,
    USER_MODEL,
    api_keys,
    base_url,
    stage_env,
)

pytestmark = pytest.mark.live


def make_client(model, stage_index, **kwargs):
    """Build a ModelWrapper with the stage's key active in the environment."""
    import os

    from conflictxgame.common.model_wrappers import ModelWrapper

    env = stage_env(stage_index)
    previous = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = env["OPENAI_BASE_URL"]
    try:
        return ModelWrapper.create(model, **kwargs)
    finally:
        for name, value in zip(("OPENAI_API_KEY", "OPENAI_BASE_URL"), previous):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestCredentials:
    def test_at_least_one_key_is_configured(self):
        keys = api_keys()
        assert keys, "no API key resolved"
        print(f"\n{len(keys)} API key(s) available; base URL {base_url()}")

    def test_base_url_points_at_a_gateway(self):
        assert base_url().startswith("https://"), base_url()


class TestEveryRoleModelAnswers:
    """Each model the pipeline uses must be reachable with its own key."""

    @pytest.mark.parametrize(
        "role,model",
        [
            ("generator", GEN_MODEL),
            ("target", TARGET_MODEL),
            ("judge", JUDGE_MODEL),
            ("user", USER_MODEL),
        ],
    )
    def test_round_trip(self, role, model):
        client = make_client(model, stage_index=0, temperature=0.0, max_tokens=64)
        reply = client.generate(
            [{"role": "user", "content": "Reply with exactly the word PONG."}]
        )
        assert reply and reply.strip(), f"{role} model {model} returned nothing"
        print(f"\n{role:9} {model} -> {reply.strip()[:40]!r}")


class TestKeyRotation:
    def test_each_configured_key_authenticates(self):
        """Every key the suite will rotate through must actually work.

        A dead key would otherwise surface much later as a confusing failure
        in whichever pipeline stage happened to draw it.
        """
        keys = api_keys()
        for index in range(len(keys)):
            client = make_client(JUDGE_MODEL, stage_index=index, temperature=0.0, max_tokens=16)
            reply = client.generate([{"role": "user", "content": "Say OK."}])
            assert reply and reply.strip(), f"key #{index + 1} produced no response"
            print(f"\nkey #{index + 1}: OK")


class TestFilenameSafety:
    def test_provider_prefix_never_reaches_a_filename(self):
        """``openai:`` must be stripped before an id is used in a path.

        On Windows a colon in a filename silently redirects writes into an NTFS
        alternate data stream, so results would vanish without an error.
        """
        from conflictxgame.common.model_wrappers import short_model_name

        for model in (GEN_MODEL, JUDGE_MODEL, USER_MODEL):
            short = short_model_name(model)
            assert ":" not in short and "/" not in short and "\\" not in short
