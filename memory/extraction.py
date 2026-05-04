"""Async memory formation via the brain LLM at config.LLM_MEMORY_URL (fire-and-forget).

Two-pass design: a thinking-OFF yes/no classifier first decides if there is
anything durable in the exchange; only on yes do we make the slow thinking-ON
extraction call. Saves substantial GPU on the common case where most exchanges
produce empty arrays.
"""

import asyncio
import json
import logging
from memory.manager import store_memory
from memory.facts import store_fact
from llm.client import generate_memory

log = logging.getLogger(__name__)

# Single-flight: drop new extraction requests while one is already running.
# Avoids GPU contention with the conversation LLM and bounds memory pressure
# under fast turn rates. Bool-flag check + set is atomic in single-threaded
# asyncio (no await between check and set).
_extraction_running = False


_CLASSIFIER_PROMPT = (
    "You are a triage step. Read this exchange and answer ONE word: yes or no.\n\n"
    "Answer YES only if the USER explicitly stated something durable about themselves\n"
    "or their world that would still be true next week: a relationship, pet, preference,\n"
    "location, occupation, project, or biographical fact. Otherwise answer NO.\n\n"
    "Examples that should be NO:\n"
    "- greetings, jokes, acknowledgments, requests, commands\n"
    "- transient states (what someone is doing right now)\n"
    "- assistant claims or guesses\n"
    "- meta-commentary about the conversation itself\n\n"
    "User: {user_text}\n"
    "Assistant: {assistant_text}\n\n"
    "Answer (one word, yes or no):"
)


async def extract_and_store(
    user_text: str,
    assistant_text: str,
    speaker_id: int | None = None,
):
    """Extract memories and facts from a conversation exchange.

    Two-pass: thinking-OFF classifier first (cheap), then thinking-ON
    structured extraction only if the classifier says yes. Single-flight:
    drops new requests while one is in progress.
    """
    global _extraction_running
    # Skip extraction for very short/empty user messages (likely STT hallucinations)
    stripped = user_text.strip()
    if len(stripped) < 15 and not any(c.isupper() for c in stripped[1:]):
        log.debug("Skipping extraction - user text too short: %r", stripped)
        return
    if _extraction_running:
        log.debug("Memory extraction skipped - previous extraction still running")
        return
    _extraction_running = True
    asyncio.create_task(_do_extraction(user_text, assistant_text, speaker_id))


async def _do_extraction(user_text: str, assistant_text: str, speaker_id: int | None):
    global _extraction_running
    try:
        # Pass 1: cheap thinking-OFF classifier
        classify_prompt = _CLASSIFIER_PROMPT.format(
            user_text=user_text, assistant_text=assistant_text
        )
        verdict = await generate_memory(classify_prompt, thinking=False)
        verdict_clean = (verdict or "").strip().lower()
        first_word = verdict_clean.split()[0].rstrip(".,!?:;") if verdict_clean else ""
        if first_word != "yes":
            log.debug("Memory classifier verdict: %r (skipping extraction)",
                      first_word or verdict_clean[:30])
            return
        log.info("Memory classifier said yes - running full extraction")

        # Pass 2: thinking-ON structured extraction
        prompt = (
            "Extract ONLY durable personal facts from this conversation exchange.\n"
            "A durable fact is something that will still be true NEXT WEEK: relationships,\n"
            "pets, preferences, locations, occupations, projects, biographical details.\n\n"
            "DO NOT extract:\n"
            "- The user's name (already known)\n"
            "- The assistant's name or anything about the assistant\n"
            "- Transient states (what someone is doing right now, testing, coming/going)\n"
            "- Conversation meta-commentary (greetings, praise, acknowledgments)\n"
            "- Anything the assistant CLAIMED or GUESSED - only facts the USER stated\n"
            "- Facts that only appear in the assistant's response, not the user's words\n"
            "- System/technical observations about the conversation itself\n"
            "- Requests or commands (e.g. 'wants Dan to recycle')\n"
            "- What someone has NOT done or hasn't said yet\n"
            "- Observations about the conversation flow or speaker behavior\n"
            "- Spelling corrections or mishearings\n"
            "- Preferences about THIS system (pause length, prompt format, etc.)\n\n"
            "CRITICAL: Only extract facts that the USER explicitly stated as personal\n"
            "biographical truth. If the user said 'I have two cats named Dexter and Preston'\n"
            "that's a durable fact. If the user said 'go recycle' that is NOT.\n\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text}\n\n"
            'Output ONLY valid JSON:\n'
            '{"facts": [{"subject": "...", "predicate": "...", "value": "..."}],\n'
            ' "memories": [{"type": "episodic|semantic", "content": "..."}]}\n\n'
            "If nothing durable, output: {\"facts\": [], \"memories\": []}\n"
            "Most exchanges have NOTHING worth storing. When in doubt, output empty arrays."
        )

        result = await generate_memory(prompt, thinking=True)
        if not result:
            return

        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(result)

        for fact_data in parsed.get("facts", []):
            subj = str(fact_data.get("subject", "")).strip()
            pred = str(fact_data.get("predicate", "")).strip()
            val = str(fact_data.get("value", "")).strip()
            if subj and pred and val:
                await store_fact(subj, pred, val, speaker_id=speaker_id)

        for mem_data in parsed.get("memories", []):
            mem_type = str(mem_data.get("type", "episodic")).strip()
            content = str(mem_data.get("content", "")).strip()
            if content and mem_type in ("episodic", "semantic", "procedural"):
                await store_memory(mem_type, content, speaker_id=speaker_id)

        log.info("Memory extraction complete: %d facts, %d memories",
                 len(parsed.get("facts", [])), len(parsed.get("memories", [])))

    except json.JSONDecodeError as e:
        log.warning("Memory extraction JSON parse failed: %s", e)
    except Exception as e:
        log.error("Memory extraction error: %s", e)
    finally:
        _extraction_running = False
