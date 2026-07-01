# Enrollment unification — Phase B + C (2026-07-01)

Unifies the three divergent enroll paths into ONE dual-modality commit and adds
a name-preserving migration/audit. Recognition already moved to okDemerzel
EdgeFace (Phase 0/A); this is the enroll side. Plan:
`~/.claude/plans/delegated-napping-cherny.md`.

## The one writer: `presence/identity_commit.py`

`commit_identity(name, *, voice_embeddings, face_crops|face_embeddings, ...)`
binds a name to a face and/or voice under ONE shared `speaker_id`
(`models/speaker/_id_map.json`). Order (S2): allocate id + upsert Postgres
`speakers` row FIRST (FK precondition), then persist voiceprint, then face
prototypes. Edge cases built in:

- **S1** one shared id space; ids never renumbered (facts/memories FKs survive).
- **S2** partial commit is valid, never an orphaned FK; partials logged.
- **S3** already-known → *augment*; a stranger cannot claim a known name
  (committing samples must match the existing identity above threshold on a
  modality that already has prototypes, else `status="mismatch"`).
- **S8** name lowercased at this single boundary.
- **S9** per-modality min-sample warnings (advisory; commits ≥1 sample).

Core (`commit_identity_stores`) is pure fs+numpy → hermetically tested
(`tests/test_identity_commit.py`). The async wrapper adds crop embedding, the
Postgres reconcile, and the shared in-memory identifier refresh (so a live
enroll recognizes immediately, no restart).

## Reactive enroll (flag-gated): `config.UNIFIED_ENROLL_ENABLED`

Default **OFF**. Enabled by EITHER env `TIMMY_UNIFIED_ENROLL=1` OR the live
runtime toggle `unified_enroll_enabled` (OR-gated; flip the toggle for no
restart). When ON, "enroll me /
remember my face / remember my voice as X" routes through `commit_identity`
(okDemerzel stores) instead of `main._handle_enrollment` (the RETIRED Pi SFace
gallery — that POST is now effectively a no-op, which is the bug this fixes).

- Scope detection: `conversation/enroll_intent.py` → `{both, face, voice}`
  (`face`/`voice` = "… my face/voice"; `both` = "enroll me" / two modalities).
  Names canonicalized lowercase (Title-case only at display).
- Keyword without a name → one-turn latch (`_pending_enroll`) asks "what name?"
  then commits on the reply.
- `main._handle_unified_enroll` pulls the passively co-sampled sole-face crops +
  the tracked unknown-voice embeddings and calls `commit_identity`.

## Passive co-sampling

`FaceObservation.sole_face_crops` is filled by `presence/face_recognize.py` ONLY
when exactly one face was detected across the grab (the sole-face==speaker rule,
`f1b95a7`), so a crop is unambiguously the speaker. `main` buffers them per
speaker in `presence/cosample.py` (`CoSampleBuffer`, bounded ring, embeds at
commit only — never per 2 Hz tick). Voice embeddings are NOT buffered here — they
already live on the tracked `UnknownSpeaker.embeddings`.

## Phase C — migration / audit: `ops/migrate_reenroll.py`

Name-preserving: migrates *names*, never vectors or ids.

    python -m ops.migrate_reenroll                       # audit table
    python -m ops.migrate_reenroll --json
    python -m ops.migrate_reenroll --reconcile-db        # upsert speakers rows (id-stable)
    python -m ops.migrate_reenroll --backup erin --modality voice   # force re-enroll, keep id/memory

Live audit 2026-07-01: `dan` complete; makers (ids 7–21) **face-only**;
`thea`/`devon`/`couples_therapist` **voice-only**; `erin` (id 5) has **neither**
vector (an id reservation whose facts FK survives). This is the fusion gap the
unified flow closes: when a face-only maker speaks at the booth (sole face +
voice), passive co-sampling + an "enroll me" (or the dual auto path) adds their
voice and completes the identity — with the same `speaker_id`.

## Deprecated (superseded, NOT deleted — Dan's call, kept while flag off)

Standalone enroll scripts, all superseded by `commit_identity` +
`ops/migrate_reenroll`:

- voice: `enroll_voiced.py`, `enroll_prototypes.py`, `enroll_voice.py`,
  `enroll_dan_v2.py`, `enroll_from_pipeline.py`, `enroll_speaker.py`,
  `ops/enroll_persona_voice.py`
- face: `enroll_face_remote.py` (Pi/HTTP), `ops/enroll_maker_faces.py` (folds
  into a `commit_identity` batch)

**Not yet retired** (gated on flipping `UNIFIED_ENROLL_ENABLED` on and
validating live): the reactive Pi POST in `main._handle_enrollment`, and the
overlapping dialogs `conversation/introductions.py` + `presence/face_enroller.py`
(the plan's "collapse into one FSM"). While the flag is OFF these remain the live
path, so they stay until the flip is proven.

## Deferred

- **Proactive dual-modality auto-enroll:** `presence/face_enroller.py`'s
  offer→capture FSM still streams to the dead Pi gallery (`_enroll_stream`). It's
  default-OFF (`TIMMY_AUTO_ENROLL_ENABLED`), so re-arming it through
  `commit_identity` is deferred (higher risk, low urgency). The reactive path
  above is the shipped dual-modality enroll.
- **Full FSM merge** of introductions + face_enroller (post-flip).

## Verify / flip checklist

1. Hermetic: `pytest tests/test_{identity_commit,enroll_intent,cosample,migrate_reenroll,face_identifier,presence,introductions}.py`.
2. Audit: `python -m ops.migrate_reenroll`.
3. Flip live: set toggle `unified_enroll_enabled=true` (or `TIMMY_UNIFIED_ENROLL=1` + restart).
4. Live: chat a few turns (warms the sole-face crop buffer) → "enroll my voice as <name>"
   → confirm `<name>_wespeaker.npy` + `<name>_edgeface.npy` share one id in
   `_id_map.json` and a `speakers` row exists; re-audit shows `complete`.
