# Self-Service Enroll — Design Proposal (DRAFT, awaiting Dan)

> Drafted by the supervisor/couples-therapist (Claude Code) 2026-06-29 while Dan
> was away, from his (truncated) voice spec: *"make a tool call — if somebody
> says 'enroll my face' / 'enroll my voice' / 'enroll me' (ideally 'enroll me'
> does everything), it does what it needs to do, with options for voice-only or
> face-only, and matches them to existing names for whoever it thinks they are.
> We also need to flip that flag."* **No code written yet — confirm before build.**

## Goal

One spoken command family, run live (no UI), self-naming:

| Utterance | Scope | Name resolution |
|-----------|-------|-----------------|
| "enroll me" / "remember me" | **both** (face + voice) | believed identity → else ask |
| "enroll my face" | face only | believed identity → else ask |
| "enroll my voice" | voice only | believed identity → else ask |
| "enroll me as Pat" / "...my name is Pat" | as given | explicit name wins |

"Match them to existing names for whoever they think it is" = if the system
already has a confident face/voice guess, enroll **under that existing name**
(re-enroll/augment), not a new identity.

## Why the regex path, not the classifier tool-router

`tool_router.py` dispatch is gated behind `classifier_enabled` (default **OFF**) and
needs grammar+prompt+branch kept in sync (3 places). The existing
`detect_enroll_intent` → `_handle_enrollment` shortcut (`main.py:487`) runs
**unconditionally** and already owns the turn on a hit. Extend *that*.

## Changes (proposed)

### 1. Broaden intent — `conversation/enroll_intent.py`
- Add `scope: "both" | "face" | "voice"` to `EnrollIntent`.
- New patterns (current gate at `:18-21` requires literal "face"):
  - `enroll|remember|learn|save|store + me`  → scope=`both`
  - `... my face`  → scope=`face`
  - `... my voice` → scope=`voice`
- Keep name extraction (`:24-30`) + speaker fallback (`:80-87`).

### 2. Unified handler — generalize `_handle_enrollment` (`main.py:271`)
`async def _handle_enrollment(name, scope, used_speaker_fallback)`:
1. **Resolve name** (priority): explicit name from utterance → else
   `verdict.face_hint_name` if `face_hint_source=="face"` → else `speaker_name`
   if not `unknown_*` → else **ask** ("Sure — what's your name?", reuse
   FaceEnroller ASK_NAME, `face_enroller.py:328`).
2. **scope ∈ {both, face}** → face enroll via existing streamerpi stream seam
   (`_enroll_stream`, `main.py:314`); name **Title-cased** for the face DB.
3. **scope ∈ {both, voice}** → voice enroll (see §3); name **lowercased**.
4. Speak one combined confirmation (`tts.speak` + `add_assistant_turn`, the
   tool-router pattern at `tool_router.py:557-560`).
5. Casing: one logical identity; face DB Title-case, voiceprint lowercase —
   fusion matches lowercase (`identity.py:19`), so they bind.

### 3. Voice enroll mechanism — depends on the flag
- **Existing known speaker** (re-enroll): `start_reenrollment(name)` — but it
  only persists if `ONLINE_LEARNING_ENABLED`.
- **New person**: `assign_name(temp_id, name)` promotes unknown→known in-memory
  (live this session) and persists **only if** `ONLINE_LEARNING_ENABLED`.
- The only writer that ignores the flag is `persist_voiceprint(name, protos)`
  (`identifier.py:437`) + prototypes built from the unknown's accumulated
  `embeddings` via `_build_prototypes` (`:198`); new `.npy` loads next restart,
  so also append an in-memory `KnownSpeaker` for the live session.

**Decision — DONE (Dan 2026-06-30):**
- `ONLINE_LEARNING_ENABLED = True` (`identifier.py:182`) — SHIPPED, deployed via
  systemd restart (started_at 2026-06-30 00:19:42). T1 (assign_name persist) +
  T2 (re-enroll) now persist durably.
- `config.SPEAKER_DRIFT_LEARNING = False` — kept off; T3 passive drift stays gated
  *separately* (`_apply_drift`, `:821`). Intentional enroll without silent drift.

## Hardware update — noise-reducing close-talk mic (Dan 2026-06-30)
Dan switched to a **plug-in noise-reducing microphone** that drastically cuts
surrounding/room noise. Two design consequences:
- **Mitigates the main flag risk.** The reason `ONLINE_LEARNING_ENABLED` was off
  was "short/noisy/off-mic clips pollute identities." Active noise reduction
  suppresses exactly that off-mic energy, so T1/T2 enroll samples are cleaner and
  off-mic bleed onto enrolled voiceprints (the `project_lt_mic_chain_misid` P1
  root) is materially reduced. The flip is safer than it would have been on the
  old lav.
- **Pipeline-match caveat (`project_speaker_enrollment` invariant).** Voiceprints
  are pipeline-matched — embeddings enrolled through the OLD front-end may match
  the NEW mic's audio worse. Existing enrolled identities (Dan, Pat voice, etc.)
  likely want **re-enrollment through the new mic** for best recognition; now
  that the flag is on, a live "enroll me" re-enroll actually persists. Also
  re-check the energy-floor / relative-amplitude addressee gate (P5) thresholds:
  a quieter noise floor shifts where off-mic gets gated out.

## Open questions for Dan
1. ~~Confirm: flip `ONLINE_LEARNING_ENABLED=True` but leave `SPEAKER_DRIFT_LEARNING=False`?~~ **DONE 6-30 — yes.**
2. Consent: "enroll me" is self-consent by the speaker — skip the FaceEnroller
   consent offer when the person explicitly asks? (Yes, I'd assume.)
3. When believed-identity is confident but the person didn't name themselves,
   confirm the match aloud first ("You're Pat, right?") or just proceed?
4. "enroll me" when a face already exists but voice is weak (today's Pat) →
   should "both" re-enroll the face too, or detect "face already solid, only do
   voice"? (Lean: enroll only what's missing/weak, report what it did.)

## Restart / deploy notes
- Voiceprint `.npy` writes load at start only; `assign_name` covers the live
  session. Plan one restart after the code lands + flag flip.
- `feedback_lt_service_restart_systemd_not_api`, `feedback_hermes_mcp_subprocess_stale_code` apply.
