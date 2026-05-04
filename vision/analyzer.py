"""Scene analysis via Qwen3.6 multimodal through llama-server.

Sends captured JPEG frames to the brain LLM at config.LLM_VISION_URL
(Qwen3.6-35B-A3B with mmproj-BF16 attached, default :8083) using the
/v1/chat/completions endpoint with image_url. Always sends
chat_template_kwargs:{enable_thinking:false} — thinking-on adds 4-13×
latency on warm images and is unnecessary for structured scene
captioning (see qwen36-vision-on-okdemerzel-2026-04-27 in Obsidian).

Returns structured JSON scene records, not free-form captions.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import httpx
import config

log = logging.getLogger(__name__)


# --- Structured scene record ---

@dataclass
class SceneRecord:
    """Structured representation of what the VLM sees."""
    timestamp: str = ""
    people: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    scene_state: str = ""
    change_from_prior: str = ""
    novelty: float = 0.0
    humor_potential: float = 0.0
    store_as_memory: bool = False
    speak_now: bool = False
    memory_tags: list[str] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for prompt injection."""
        parts = []
        if self.people:
            parts.append("People: " + ", ".join(self.people))
        if self.actions:
            parts.append("Activity: " + ", ".join(self.actions))
        if self.objects:
            parts.append("Objects: " + ", ".join(self.objects[:5]))
        if self.scene_state:
            parts.append("Scene: " + self.scene_state)
        if self.change_from_prior and self.change_from_prior.lower() != "none":
            parts.append("Changed: " + self.change_from_prior)
        return "; ".join(parts) if parts else "Nothing notable"


# --- VLM prompt for structured output ---
# This is the SINGLE source of truth for the visual prompt. Any future change
# to what the model is asked to do with an image (different fields, different
# instructions, different style) goes here — every caller in vision/ delegates
# to analyze_frame() which uses this constant by default.

STRUCTURED_PROMPT = (
    "Analyze this camera frame and return ONLY a JSON object with these fields:\n"
    "{\n"
    '  "people": ["list of people visible, use names if recognizable or '
    "descriptors like 'person in black jacket'\"],\n"
    '  "objects": ["notable objects in active use or prominently visible"],\n'
    '  "actions": ["what people are doing in task-oriented terms"],\n'
    '  "scene_state": "brief phrase describing the overall scene",\n'
    '  "change_from_prior": "what looks like it changed recently, or none if static",\n'
    '  "novelty": 0.0 to 1.0 how unusual or noteworthy this scene is,\n'
    '  "humor_potential": 0.0 to 1.0 how comment-worthy or funny the situation is,\n'
    '  "store_as_memory": true if this scene is worth remembering long-term,\n'
    '  "speak_now": true only if something surprising or urgent is happening,\n'
    '  "memory_tags": ["category:value tags like activity:soldering, location:workbench"]\n'
    "}\n"
    "Return ONLY valid JSON. No explanation, no markdown fences."
)

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
    return _client


def _parse_scene_json(text: str) -> dict | None:
    """Try to extract JSON from VLM response, handling common issues."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


async def analyze_frame(jpeg_bytes: bytes, prompt: str | None = None) -> SceneRecord | None:
    """Send a JPEG frame to the brain VLM and get a structured scene record.

    Uses /v1/chat/completions with image_url (data URI). Sends
    chat_template_kwargs:{enable_thinking:false} so Qwen3.6 emits the
    JSON answer directly without a thinking trace.

    Args:
        jpeg_bytes: Raw JPEG image data.
        prompt: Optional custom prompt. Defaults to STRUCTURED_PROMPT.

    Returns:
        SceneRecord with structured scene data, or None on failure.
    """
    client = await _get_client()
    b64_image = base64.b64encode(jpeg_bytes).decode("ascii")

    payload = {
        "model": config.LLM_BRAIN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + b64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt or STRUCTURED_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        t0 = time.monotonic()
        resp = await client.post(
            config.LLM_VISION_URL + "/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - t0

        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        tokens = usage.get("completion_tokens", 0)

        log.info("[VISION] %.1fs, %d tokens: %s", elapsed, tokens, content[:120])

        # Parse JSON response into SceneRecord
        parsed = _parse_scene_json(content)
        if parsed is None:
            log.warning("[VISION] Failed to parse JSON, falling back to raw caption")
            return SceneRecord(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                scene_state=content[:200],
                novelty=0.5,
            )

        record = SceneRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            people=parsed.get("people", []),
            objects=parsed.get("objects", []),
            actions=parsed.get("actions", []),
            scene_state=parsed.get("scene_state", ""),
            change_from_prior=parsed.get("change_from_prior", ""),
            novelty=float(parsed.get("novelty", 0.0)),
            humor_potential=float(parsed.get("humor_potential", 0.0)),
            store_as_memory=bool(parsed.get("store_as_memory", False)),
            speak_now=bool(parsed.get("speak_now", False)),
            memory_tags=parsed.get("memory_tags", []),
            raw_json=parsed,
        )

        log.info("[VISION] Scene: %s", record.summary())
        return record

    except httpx.ConnectError:
        log.debug("Vision server not reachable at %s", config.LLM_VISION_URL)
        return None
    except httpx.HTTPStatusError as e:
        log.warning("Vision analysis HTTP error: %s", e)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        log.warning("Vision analysis parse error: %s", e)
        return None
    except Exception:
        log.exception("Vision analysis failed")
        return None


async def check_model_available() -> bool:
    """Check if the brain VLM server is responding."""
    client = await _get_client()
    try:
        resp = await client.get(config.LLM_VISION_URL + "/health")
        if resp.status_code == 200:
            return True
        log.warning("Vision server health check returned %d", resp.status_code)
        return False
    except httpx.ConnectError:
        log.warning(
            "Vision server not reachable at %s — qwen36-server.service should be active.",
            config.LLM_VISION_URL,
        )
        return False
    except Exception:
        log.warning("Could not check vision server health")
        return False
