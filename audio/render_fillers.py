"""One-time renderer: freeze the FILLERS tuple to committed .wav assets.

Run whenever you edit the FILLERS tuple (or change the Piper voice) to
regenerate the locked filler clips that LT loads at startup:

    .venv/bin/python -m audio.render_fillers

Audio is synthesized exactly the way the runtime used to cache it
(_synthesize_raw, no comma substitution — commas stay as natural Piper
pauses), so the frozen .wav is bit-for-bit what the old in-RAM prewarm
produced. Files are content-addressed by filler text (audio.fillers.wav_path),
so stale clips from removed/edited phrases are pruned automatically.
"""

import logging

import soundfile as sf

import config
from audio import fillers
from tts.engine import _synthesize_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("render_fillers")


def main() -> None:
    fillers.WAV_DIR.mkdir(parents=True, exist_ok=True)

    keep = set()
    for text in fillers.FILLERS:
        path = fillers.wav_path(text)
        keep.add(path.name)
        audio, sr = _synthesize_raw(text, config.PIPER_MODEL)
        if len(audio) == 0:
            log.warning("EMPTY synth, skipped: %r", text)
            continue
        # float32 WAV preserves the Piper float output exactly; sr is stored in
        # the header so the loader needs no out-of-band sample-rate info.
        sf.write(path, audio, sr, subtype="FLOAT")
        log.info("wrote %s  (%5.2fs)  %r", path.name, len(audio) / sr, text)

    # Prune clips whose source phrase no longer exists in the tuple.
    for stale in fillers.WAV_DIR.glob("filler_*.wav"):
        if stale.name not in keep:
            stale.unlink()
            log.info("pruned stale %s", stale.name)

    log.info("done: %d filler clips in %s", len(keep), fillers.WAV_DIR)


if __name__ == "__main__":
    main()
