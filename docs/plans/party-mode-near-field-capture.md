# Plan: Party-Mode Near-Field Capture + VU Meter (LT-OS)

**Goal:** Stop Little Timmy from conversing with background voices at a crowded party
(busy-conference acoustics) when Dan is speaking into a body-worn wireless lavalier.
Add operator-facing knobs on LT-OS (incl. a live VAD-threshold control) and a simple
VU meter (current volume + peak-hold) so the floors can be calibrated live.

**Author:** Demerzel session 2026-06-09. **Status:** proposed.

---

## Why this works

A lavalier sits on the body, so Dan's direct voice peaks **far** higher than anyone
across the room. Two independent filters, layered:

1. **Energy floor (near-field gate)** — reject onset below a peak/RMS threshold. Cheap,
   runs before STT, kills most distant chatter. This is the user's "filter by volume" idea
   and it's well-suited to a lav.
2. **Speaker allowlist (identity gate)** — the Resemblyzer voiceprint infra already exists
   (Dan enrolled, threshold 0.30); today `main.py:340` only discards *Timmy's* voice. Gate
   replies to an enrolled allowlist. Catches the *loud-but-wrong* guest who leans into the mic.

Plus tuning: raise `VAD_THRESHOLD` so marginal/distant speech doesn't trip onset.

Everything ships **behind flags, default OFF** → zero regression to normal operation.

---

## Architecture (confirmed in code)

| Concern | File | Hook |
|---|---|---|
| Capture / VAD / onset | `audio/capture.py` | `is_speech` check (~L233), onset block (~L268), `diag_last_peak` (L237), `set_hearing()` pattern (L382) |
| Static threshold | `config.py:36` | `VAD_THRESHOLD = 0.4` (hardcode → live knob) |
| Speaker gate | `main.py:340` (`is_timmy` discard) + `speaker/identifier.py` (`KNOWN_SPEAKER_THRESHOLD=0.30`) |
| Persisted knobs | `persistence/runtime_toggles.py` | `_DEFAULTS` + `get/set` |
| LT control API | `web/app.py:8893` | `broadcast_event()` (L42), `/api/hearing` (L572), `/api/audio/diag` (L385) |
| LT-OS dashboard | `little_timmy_os/main.py:8894` + `services.py` | WS `lt_toggles` dispatch (L149-171), inline HTML at `/` |

---

## Phase 0 — Persisted knobs (foundation)

Add to `persistence/runtime_toggles.py` `_DEFAULTS`:

```python
"capture_vad_threshold": 0.4,        # Silero onset prob floor (live-tunable)
"capture_energy_floor": 0.0,         # peak amplitude floor for onset; 0.0 == disabled
"party_mode_enabled": False,         # master switch for near-field/allowlist gating
"speaker_allowlist": ["dan"],        # names that may get a reply when party_mode on; [] == allow all
```

`runtime_toggles.get/set` already handle arbitrary JSON types and re-read on every `get()`,
so a manual file edit or a UI write both take effect without restart. (One caveat: `_load()`
type-guards against the *default's* type — a list default is fine, verify list round-trips.)

---

## Phase 1 — Energy floor + live VAD knob (the volume answer)

**`audio/capture.py`:**

1. In `__init__`, read knobs into instance attrs (mirror `hearing_muted`):
   ```python
   self.vad_threshold   = float(_toggles.get("capture_vad_threshold"))
   self.energy_floor    = float(_toggles.get("capture_energy_floor"))
   ```
2. Replace `is_speech = vad_prob >= config.VAD_THRESHOLD` with:
   ```python
   peak = float(np.max(np.abs(audio)))          # already computed for diag at L237 — reuse
   is_speech = (vad_prob >= self.vad_threshold) and (peak >= self.energy_floor)
   ```
   Energy floor `0.0` ⇒ no-op (default safe). Apply the floor to **onset only** (the `not recording`
   branch ~L268), NOT to the mid-utterance/silence-count path, so a momentarily quieter syllable
   mid-sentence doesn't prematurely endpoint.
3. Add setters mirroring `set_hearing()` (update attr + `runtime_toggles.set` + log):
   `set_vad_threshold(v)`, `set_energy_floor(v)`.

**`config.py`:** keep `VAD_THRESHOLD = 0.4` as the seed default only; capture now owns the live value.

**Why peak (max-abs) over RMS:** peak is already computed, and for a lav the transient peak
separates near/far better than slow RMS. (If peak proves jumpy in testing, switch the floor to a
short rolling RMS — the DWU `pcm_meansq` pattern from `feedback_wake_word_acoustic_gates`.)

---

## Phase 2 — Speaker allowlist (identity answer)

**`main.py`**, immediately after the `is_timmy` discard (~L344, before STT):

```python
if runtime_toggles.get("party_mode_enabled"):
    allow = [n.lower() for n in runtime_toggles.get("speaker_allowlist") or []]
    nм = speaker_result.name.lower()
    if allow and (nм.startswith("unknown") or nм not in allow):
        log.debug("Party mode: dropping non-allowlisted speaker %s (conf=%.2f)",
                  speaker_result.name, speaker_result.confidence)
        return
```

- Gate sits **after** speaker ID, **before** STT/name-solicitation, so unknown party guests get
  neither a reply nor a "what's your name?" prompt.
- `speaker_allowlist == []` ⇒ allow-all (party mode still gives you the energy floor + VAD bump
  without identity filtering, useful if voiceprint proves flaky on the lav).
- **Known caveat** (from `project_speaker_enrollment`): 5 kHz lav + short/overlapping utterances
  blur embeddings; if a guest co-speaks into Dan's mic the blended segment can drift past 0.30 and
  reject Dan. That's why this is layered with the energy floor, not solo.

**Pre-party task:** re-verify Dan's voiceprint *through the lavalier path* (`enroll_from_pipeline.py`
— enrollment must match live capture path, the proven lesson). The existing enrollment may have
been via the okDemerzel desk mic, not the lav.

---

## Phase 3 — LT-OS UI: VAD slider + VU meter

### 3a. VAD threshold knob (+ energy floor + party-mode toggle)

**LT (`web/app.py`):** add `GET/POST /api/capture/config` mirroring `/api/hearing`:
```python
GET  -> {"vad_threshold": .., "energy_floor": .., "party_mode": .., "allowlist": [..]}
POST -> applies via capture setters + runtime_toggles, returns new state
```

**LT-OS (`services.py` + `main.py`):**
- `services.get_capture_config()` / `set_capture_config(**kw)` → httpx to `:8893/api/capture/config`
  (mirror `check_lt_toggles_status`/`toggle_hearing`).
- Extend the WS `toggle` dispatch (main.py L149-171) with a `capture_config` message type carrying
  `{vad_threshold, energy_floor, party_mode}`; broadcast updated state back like `lt_toggles`.
- Frontend: a **range slider** (0.0–1.0, step 0.05) for VAD threshold, a slider for energy floor,
  and a party-mode toggle in the existing toggles panel. On `input`/`change`, send the WS message
  (debounce ~150ms). Show the live numeric value beside each slider.

### 3b. VU meter (simple, peak-hold)

**Data source already exists:** `GET :8893/api/audio/diag` returns `last_peak` (+ `last_vad_prob`).

**Simplest version (recommended, ~zero backend):**
- LT-OS proxy route `GET /api/audio/diag` → forwards to `:8893/api/audio/diag`.
- Frontend `setInterval(~150ms)` fetch; render two stacked CSS bars:
  - **Current volume** — width = `last_peak` (0–1), green→amber→red gradient.
  - **Peak-hold** — a thin marker at the max of `last_peak` over a sliding ~2s window, decaying
    ~3 dB/s when no new peak exceeds it (classic VU peak-hold; computed client-side).
- Tint the bar by `last_vad_prob ≥ threshold` (e.g., border glows when capture considers it speech)
  so Dan can *see* the floor doing its job: his voice lights the bar, room chatter stays dark/short.

**Optional accuracy upgrade (if 150ms polling looks coarse):** add `diag_peak_hold` to capture
(max with per-chunk decay at the ~30–60 Hz chunk rate) and expose it in `/api/audio/diag`; frontend
just renders it. Keeps peak-hold accurate without faster polling.

**Calibration UX:** with the meter live, Dan reads off two numbers at the party — his speaking peak
vs. ambient/background peak — and drags the energy-floor slider to sit between them. The VAD glow
confirms onset only fires on his voice. This is the whole point of shipping the meter with the knob.

---

## Rollout / safety

- **Defaults OFF**: `party_mode_enabled=False`, `energy_floor=0.0` (no-op), `vad_threshold=0.4`
  (current behavior). Nothing changes until Dan flips party mode / drags a slider.
- **All persisted** via `runtime_toggles` → survives `little-timmy.service` restart; reset to defaults
  by deleting `data/lt_runtime_toggles.json` keys.
- **Single-flip Saturday:** party-mode toggle on LT-OS enables the allowlist gate; sliders tune floors
  live. Toggle off after the party.
- **Diagnostics:** existing `[DIAG] peak=.. vad=..` log line (capture.py:242) already prints what the
  meter shows — cross-check there if the UI looks wrong.

## Test plan

- **Offline:** unit-test the onset predicate (`vad_prob`/`peak`/floor truth table); test the allowlist
  gate (timmy / dan / unknown / empty-allowlist) — pure functions, no audio.
- **Live (pre-party):** play background-chatter audio across the room + speak into the lav; confirm
  (a) meter separates the two, (b) energy floor at the calibrated value drops the chatter, (c) party
  mode drops a non-enrolled helper's voice even when loud. Live-test with **real phrasing**, not the
  unit suite (per `feedback_live_test_classifiers_with_real_phrasing`).
- Re-enroll Dan through the lav path first (Phase 2 task) and confirm distances land <0.30.

## Sequencing

Phase 0 → 1 give the volume fix + live VAD knob (smallest, highest value — ship first).
Phase 3b (VU meter) can land right after Phase 1 since `/api/audio/diag` already exposes peak.
Phase 2 (allowlist) is independent; ship if the energy floor alone proves insufficient in testing.
