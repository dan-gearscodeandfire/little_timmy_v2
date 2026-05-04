"""Verify Qwen3.6 fact-extraction path on :8083 with thinking=True.

The user's stated requirement for this migration was: "ensure response
does not land in reasoning_content". The server-side fix is
`--reasoning-format deepseek` on qwen36-server.service. These tests
guard that fix from regressing and confirm Little Timmy's extraction
pipeline still produces clean facts.

Run from repo root:
    .venv/bin/pytest tests/test_qwen36_brain.py -v
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

# Make the Little Timmy package importable when pytest is run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from llm.client import generate_memory  # noqa: E402


# ---------- Probe-shape test (the regression-test the user asked for) ----------

@pytest.mark.asyncio
async def test_thinking_on_answer_lands_in_content():
    """With --reasoning-format deepseek, thinking=True must put the final
    answer in `content` and only the trace in `reasoning_content`."""
    body = {
        "messages": [{"role": "user", "content": "What is 2+2? One word."}],
        # 1500 tokens gives ample headroom for the thinking trace — Qwen3.6
        # thinking length is variable even at temperature=0 (depends on KV
        # cache state), and a too-tight cap is the difference between this
        # test passing alone and failing in a suite run. The thing we're
        # verifying is "answer lands in content", not "thinking is short".
        "max_tokens": 1500,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{config.LLM_MEMORY_URL}/v1/chat/completions", json=body)
        r.raise_for_status()
        payload = r.json()
        msg = payload["choices"][0]["message"]
        finish = payload["choices"][0]["finish_reason"]

    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    assert finish == "stop", (
        f"unexpected finish_reason {finish!r} — bump max_tokens or check server. "
        f"reasoning_content was: {reasoning[:200]!r}"
    )
    assert content.strip(), (
        f"content is empty — server is not splitting thinking. "
        f"Check `--reasoning-format deepseek` on qwen36-server.service. "
        f"reasoning_content was: {reasoning[:200]!r}"
    )
    assert "<think>" not in content, (
        f"<think> tag leaked into content: {content!r}"
    )
    assert "Four" in content or "four" in content.lower(), (
        f"unexpected answer: {content!r}"
    )


@pytest.mark.asyncio
async def test_thinking_off_baseline():
    """Sanity: thinking=False keeps the answer in content with no reasoning."""
    body = {
        "messages": [{"role": "user", "content": "What is 2+2? One word."}],
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{config.LLM_MEMORY_URL}/v1/chat/completions", json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]

    assert (msg.get("content") or "").strip()
    assert not (msg.get("reasoning_content") or "").strip()


# ---------- Fixture conversation: end-to-end through generate_memory() ----------

def _build_extraction_prompt(user: str, asst: str) -> str:
    return (
        "Extract ONLY durable personal facts from this conversation exchange.\n"
        "A durable fact is something that will still be true NEXT WEEK: "
        "relationships, pets, preferences, locations, occupations, projects, "
        "biographical details.\n\n"
        "DO NOT extract: transient states, conversation meta-commentary, "
        "system observations, roleplay language.\n\n"
        f"User: {user}\nAssistant: {asst}\n\n"
        'Output ONLY valid JSON of shape:\n'
        '{"facts": [{"subject": "...", "predicate": "...", "value": "..."}], '
        '"memories": []}\n\n'
        'If nothing durable, output: {"facts": [], "memories": []}'
    )

FIXTURE_TURNS = [
    ("My dog is a corgi named Wally.", "Cute. He'd probably like you better than I do."),
    ("I work at Anthropic.", "An AI company. So you built me to spite your day job?"),
    ("My wife's name is Erin, spelled E-R-I-N.", "Noted. I'll try not to butcher it."),
    ("I live in Montclair, New Jersey.", "Sounds suburban. I'm sure that's thrilling."),
    ("I'm allergic to shellfish.", "One more thing the world has against you."),
]


@pytest.mark.asyncio
async def test_fixture_conversation_extracts_durable_facts():
    """Run the real extraction prompt through generate_memory() with
    thinking=True. Assert >=3 of the 5 durable facts are extracted and
    no obvious meta/roleplay garbage leaks through."""
    extracted_facts: list[dict] = []
    extracted_blobs: list[str] = []

    for user, asst in FIXTURE_TURNS:
        prompt = _build_extraction_prompt(user, asst)
        raw = await generate_memory(prompt, thinking=True)
        extracted_blobs.append(raw)

        # Same JSON-cleanup the production extractor does
        s = raw.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            continue
        for f in parsed.get("facts", []) or []:
            extracted_facts.append(f)

    # No <think> tag bleed into any returned blob (server-side split working)
    for blob in extracted_blobs:
        assert "<think>" not in blob, f"<think> leaked: {blob[:200]!r}"

    # The five durable facts in the fixture, by subject keyword
    expected_subjects = ["dog", "anthropic", "erin", "montclair", "shellfish"]
    flat = " | ".join(
        f"{f.get('subject','')} {f.get('predicate','')} {f.get('value','')}".lower()
        for f in extracted_facts
    )
    hits = sum(1 for kw in expected_subjects if kw in flat)
    assert hits >= 3, (
        f"expected at least 3 of 5 durable facts, got {hits}.\n"
        f"raw extraction blobs:\n  " + "\n  ".join(b[:160] for b in extracted_blobs)
    )

    # No obvious garbage from the known issues note (assistant self-descriptions,
    # roleplay, transient states):
    forbidden_substrings = ["roleplay", "tortoise", "darkness", "performing"]
    for f in extracted_facts:
        text = " ".join(str(f.get(k, "")) for k in ("subject", "predicate", "value")).lower()
        for bad in forbidden_substrings:
            assert bad not in text, f"garbage fact leaked: {f}"
