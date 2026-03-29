"""Async memory formation via GPT-OSS-120B (fire-and-forget)."""

import asyncio
import json
import logging
from memory.manager import store_memory
from memory.facts import store_fact
from llm.client import generate_memory

log = logging.getLogger(__name__)

# Only one extraction at a time to avoid GPU contention with conversation LLM
_semaphore = asyncio.Semaphore(1)


async def extract_and_store(
    user_text: str,
    assistant_text: str,
    speaker_id: int | None = None,
):
    """Extract memories and facts from a conversation exchange.

    Runs on GPT-OSS-120B (port 8080). Fire-and-forget — non-critical.
    Uses semaphore to prevent GPU contention with conversation LLM.
    """
    # Skip extraction for very short/empty user messages (likely STT hallucinations)
    stripped = user_text.strip()
    if len(stripped) < 15 and not any(c.isupper() for c in stripped[1:]):
        log.debug("Skipping extraction - user text too short: %r", stripped)
        return
    if not _semaphore.locked():
        asyncio.create_task(_do_extraction(user_text, assistant_text, speaker_id))
    else:
        log.debug("Memory extraction skipped — previous extraction still running")


async def _do_extraction(user_text: str, assistant_text: str, speaker_id: int | None):
    async with _semaphore:
        try:
            prompt = (
                "Extract ONLY durable personal facts from this conversation exchange.\n"
                "A durable fact is something that will still be true NEXT WEEK: relationships,\n"
                "pets, preferences, locations, occupations, projects, biographical details.\n\n"
                "DO NOT extract:\n"
                "- The user's name (already known)\n"
                "- The assistant's name or anything about the assistant\n"
                "- Transient states (what someone is doing right now, testing, coming/going)\n"
                "- Conversation meta-commentary (greetings, praise, acknowledgments)\n"
                "- Anything the assistant CLAIMED or GUESSED — only facts the USER stated\n"
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

            result = await generate_memory(prompt)
            if not result:
                return

            # Try to parse JSON from the response
            # Handle potential markdown code blocks
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
