"""Enroll a guest's VOICE and link it to the faceprint Timmy already has.

Built 2026-08-18 after an audit found the real gap in the identity stores:
**37 faceprints, 2 voiceprints.** Everyone met at Open Sauce (Allen Pan, Simone
Giertz, William Osman, Colin Furze, Erin, Pat, Dave...) is known by FACE only.
Nothing was stale and nothing needed re-enrolling — those voices were simply
never captured. This is the "next time they're in the shop" flow.

HOW THE LINK ACTUALLY WORKS
  There is no join table. The three stores are keyed by the same canonical
  name string:

      models/face/<name>_edgeface.npy       face
      models/speaker/<name>_wespeaker.npy   voice
      models/speaker/_id_map.json           name -> speaker_id (source of truth)
      postgres speakers(id, name)           FK target for facts/memories

  So enrolling a voice under EXACTLY the faceprint's name IS the link. Get the
  spelling wrong and you silently create a second person who happens to look
  like the first. That is the whole reason this script does a pre-flight and
  refuses to guess: it will suggest near-matches, but never pick one for you.

WHY NOT enroll_voiced.py DIRECTLY
  That script's POSES sweep distance and off-axis angle ("step back a few feet",
  "turned away from the mic"). Those date from the lav era and are now
  **anti-aligned with the close-talk design invariant** (Dan, 2026-06-17): LT
  should only ever address someone speaking into a mic they hold or wear, and
  off-mic audio is meant to be GATED OUT before the matcher, not learned. Teaching
  a guest's voiceprint to match their off-mic voice is precisely the P1 misID root
  — it is how off-mic guests collapse onto enrolled identities. So this uses
  close-mic cues only, and varies CONTENT instead of distance.

  It also follows the 2026-08-13 lesson: collect SHORT, SEPARATE sentences with
  pauses. A monologue dedupes down to a single sample (cap is 12, dedup at 0.06),
  so one long ramble is worth one clip.

USAGE
  # who needs a voice? (no audio, no writes)
  ./.venv/bin/python enroll_guest.py --list

  # check before committing anyone to it
  ./.venv/bin/python enroll_guest.py allen_pan --dry-run

  # do it — Timmy speaks the cues, guest just talks
  ./.venv/bin/python enroll_guest.py allen_pan

Restart little-timmy afterwards to load the new print.
"""

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.spatial.distance import cosine

FACE_DIR = REPO / "models" / "face"
VOICE_DIR = REPO / "models" / "speaker"

# Close-mic only, content varied instead of distance. Each cue asks for ONE
# short sentence so every clip is a distinct sample rather than a slice of one
# monologue (see the 2026-08-13 dedup lesson in the module docstring).
CUES = [
    "Say your first name, then tell me where you're from.",
    "Tell me one thing you have built or made recently.",
    "Count from one to five for me, at a normal pace.",
    "Say this back to me: the quick brown fox jumps over the lazy dog.",
    "Tell me what you had for breakfast today.",
    "Say something you would say to a friend, one sentence.",
    "Tell me the last thing that annoyed you.",
    "Last one. Say anything you like, one sentence.",
]

INTRO = ("I am going to learn your voice. Stay close to the microphone the whole "
         "time, and answer each thing I ask with one short sentence, then stop.")


def _stem(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def inventory() -> dict:
    face = {_stem(p, "_edgeface.npy") for p in FACE_DIR.glob("*_edgeface.npy")}
    voice = {_stem(p, "_wespeaker.npy") for p in VOICE_DIR.glob("*_wespeaker.npy")
             if ".bak" not in p.name}
    ids = {}
    idmap = VOICE_DIR / "_id_map.json"
    if idmap.exists():
        try:
            ids = json.loads(idmap.read_text())
        except Exception:                                    # noqa: BLE001
            ids = {}
    return {"face": face, "voice": voice, "ids": ids}


def cmd_list() -> int:
    inv = inventory()
    face, voice = inv["face"], inv["voice"]
    both = sorted(face & voice)
    face_only = sorted(face - voice)
    voice_only = sorted(voice - face)
    print(f"faceprints: {len(face)}   voiceprints: {len(voice)}\n")
    print(f"FACE + VOICE ({len(both)}) — fully enrolled, nothing to do")
    print(f"  {', '.join(both) or '(none)'}\n")
    print(f"FACE ONLY ({len(face_only)}) — Timmy recognises the face, not the voice.")
    print("  Grab these when they are next in the shop:")
    for i in range(0, len(face_only), 4):
        print("    " + ", ".join(face_only[i:i + 4]))
    if voice_only:
        print(f"\nVOICE ONLY ({len(voice_only)}) — no faceprint: {', '.join(voice_only)}")
    print(f"\n  ./.venv/bin/python enroll_guest.py <name>        # enroll one")
    return 0


def preflight(name: str, inv: dict) -> tuple[bool, str | None]:
    """Report what exists for `name` and whether it is safe to proceed.

    Returns (ok_to_proceed, resolved_name). Never auto-corrects a name — a
    near-miss silently forks the identity, so the human confirms the spelling.
    """
    face, voice, ids = inv["face"], inv["voice"], inv["ids"]
    has_face, has_voice = name in face, name in voice

    print(f"\n--- pre-flight: {name!r} ---")
    print(f"  faceprint     : {'YES' if has_face else 'no'}"
          f"{'  -> voice will link to it by name' if has_face else ''}")
    print(f"  voiceprint    : {'YES (will be REPLACED, old file backed up)' if has_voice else 'no'}")
    print(f"  speaker id    : {ids.get(name, '(assigned on first enrol)')}")

    if not has_face:
        close = difflib.get_close_matches(name, sorted(face), n=4, cutoff=0.6)
        if close:
            print(f"\n  !! No faceprint called {name!r}, but these look close:")
            for c in close:
                print(f"       {c}")
            print("  Enrolling under a NEW name creates a SECOND identity for the same\n"
                  "  person — Timmy would know their face and voice as different people.\n"
                  "  Re-run with the exact name above, or continue only if this really is\n"
                  "  someone new.")
            return False, None
        print("\n  No faceprint — this will be a voice-only identity. Fine for someone\n"
              "  Timmy has never seen; if he HAS seen them, enrol the face first so the\n"
              "  two stores share a name.")
    return True, name


def collision_check(name: str, protos: np.ndarray, inv: dict) -> None:
    """Warn if the new print sits close to an existing identity.

    A guest whose print lands inside someone else's radius is the lookalike
    failure mode: from then on either can be attributed to the other, and the
    facts stores diverge. Cheaper to catch here than after a week of
    cross-attributed memories.
    """
    from speaker.identifier import KNOWN_SPEAKER_THRESHOLD
    worst = []
    for other in sorted(inv["voice"] - {name}):
        try:
            emb = np.load(VOICE_DIR / f"{other}_wespeaker.npy")
        except Exception:                                    # noqa: BLE001
            continue
        emb = np.atleast_2d(emb)
        d = min(cosine(a, b) for a in np.atleast_2d(protos) for b in emb)
        worst.append((d, other))
    if not worst:
        return
    worst.sort()
    d, other = worst[0]
    print(f"\n  nearest existing voice: {other} at distance {d:.3f} "
          f"(match threshold {KNOWN_SPEAKER_THRESHOLD})")
    if d < KNOWN_SPEAKER_THRESHOLD:
        print(f"  !! WARNING: closer than the match threshold. Timmy may confuse\n"
              f"     {name} and {other}. Re-enrol with more varied sentences, or\n"
              f"     expect cross-attribution and watch for it.")
    else:
        print("  clear of the threshold — no collision.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="canonical name, e.g. allen_pan")
    ap.add_argument("--list", action="store_true", help="who has a face but no voice")
    ap.add_argument("--dry-run", action="store_true", help="pre-flight only, no audio")
    ap.add_argument("--clips", type=int, default=len(CUES),
                    help=f"how many cues to run (default {len(CUES)}, max {len(CUES)})")
    args = ap.parse_args()

    if args.list or not args.name:
        return cmd_list()

    name = args.name.strip().lower().replace(" ", "_").replace("-", "_")
    inv = inventory()
    ok, name = preflight(name, inv)
    if not ok:
        return 1
    if args.dry_run:
        print("\n  --dry-run: stopping before audio.")
        return 0

    # Imported late so --list / --dry-run need no audio stack or running LT.
    import enroll_voiced as ev
    from speaker.identifier import SpeakerIdentifier, _build_prototypes

    cues = CUES[:max(1, min(args.clips, len(CUES)))]
    print(f"\n  {len(cues)} clips, close-mic. Timmy speaks each cue; the guest answers\n"
          f"  with ONE short sentence and stops. Hearing is muted throughout and\n"
          f"  restored at the end.\n")

    ident = SpeakerIdentifier()
    raw, misses = [], 0
    try:
        ev.set_hearing(False)
        ev.announce(INTRO)
        for i, cue in enumerate(cues, 1):
            print(f"  [{i}/{len(cues)}] {cue}")
            ev.announce(cue)
            clip = ev.capture_16k(ev.CLIP_SECONDS)
            peak = float(np.max(np.abs(clip))) if clip.size else 0.0
            if peak < ev.MIN_PEAK:
                print(f"        too quiet (peak {peak:.3f}) — retrying once")
                ev.announce("I did not catch that. Say it again, closer to the microphone.")
                clip = ev.capture_16k(ev.CLIP_SECONDS)
                peak = float(np.max(np.abs(clip))) if clip.size else 0.0
            if peak < ev.MIN_PEAK:
                misses += 1
                print(f"        skipped (peak {peak:.3f})")
                continue
            raw.append(ev.embed(clip))
            print(f"        captured (peak {peak:.3f})")
        ev.announce("Got it. Thank you.")
    finally:
        # Hearing must come back even if this dies mid-run, or LT stays deaf.
        ev.set_hearing(True)

    if not raw:
        print("\n  no usable clips — nothing written. Check mic gain and try again.")
        return 1

    protos = _build_prototypes(raw)
    out = ident.persist_voiceprint(name, protos)
    print(f"\n  saved {protos.shape[0]} prototype(s) -> {out}")
    print(f"  captured {len(raw)}/{len(cues)} clips ({misses} skipped)")

    collision_check(name, protos, inv)

    try:
        from db.speakers import ensure_rows_for_enrolled
        n = ensure_rows_for_enrolled()
        print(f"  synced speakers table ({n} new row(s))")
    except Exception as e:                                   # noqa: BLE001
        print(f"  WARNING: speakers sync failed ({e}); a restart will reconcile it.")

    if name in inv["face"]:
        print(f"  LINKED: models/face/{name}_edgeface.npy + models/speaker/{name}_wespeaker.npy")
    print("\n  Restart little-timmy to load it:  sudo -n systemctl restart little-timmy.service")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
