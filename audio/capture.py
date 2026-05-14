"""Microphone capture with Silero VAD and hybrid endpointing.

Always-listening audio capture that:
1. Records from USB mic at native rate (typically 48kHz), downsamples to 16kHz
2. Runs Silero VAD on each chunk (~1ms per call)
3. Uses hybrid endpointing: fast finalize (0.3s) for complete sentences,
   patient wait (1.5s) for incomplete fragments
4. Emits complete speech segments (16kHz float32) to an asyncio queue
"""

import asyncio
import collections
import logging
import queue
import threading
import time
import numpy as np
import config

log = logging.getLogger(__name__)

NATIVE_RATE = 48000  # USB mic native sample rate


def looks_complete(text: str) -> bool:
    """Check if text looks like a complete sentence (ends with terminal punctuation)."""
    if not text:
        return False
    return text.rstrip()[-1] in ".?!"


class AudioCapture:
    def __init__(self):
        self.speech_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.suppressed = False  # set True during TTS playback to prevent feedback
        # User-facing mute. Same gate semantics as `suppressed` (frames go
        # through VAD for diagnostics, then get dropped before STT) so the
        # whisper-server stays untouched and unmuting is instant.
        # Bundle D 2026-05-14: read persisted state instead of
        # hard-coding unmuted so a manual mute survives a reboot.
        from persistence import runtime_toggles as _toggles
        self.hearing_muted = not _toggles.get("hearing_enabled")
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # VAD model loaded lazily
        self._vad_model = None
        # Live-transcription worker (offloads HTTP off the capture thread)
        self._live_audio_queue: queue.Queue | None = None
        self._latest_live_text: str = ""
        self._live_worker_thread: threading.Thread | None = None
        # Diagnostics
        self.diag_chunks_processed = 0
        self.diag_last_peak = 0.0
        self.diag_last_vad_prob = 0.0
        self.diag_overflows = 0
        self.diag_device_name = "unknown"

    def _load_vad(self):
        """Load Silero VAD model."""
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._vad_model = model
        log.info("Silero VAD loaded")

    def _check_vad(self, audio_chunk: np.ndarray) -> float:
        """Run VAD on a 512-sample chunk. Returns speech probability."""
        import torch
        tensor = torch.from_numpy(audio_chunk).float()
        with torch.no_grad():
            prob = self._vad_model(tensor, config.SAMPLE_RATE).item()
        return prob

    def _find_mic(self) -> int | None:
        """Find the best microphone device.

        Preference order:
        1. Onboard HD-Audio analog input (SN6186 — may appear via PipeWire)
        2. USB mic device
        3. PipeWire/PulseAudio default (routes to system default source)
        4. None (sounddevice falls back to its own default)
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            onboard = None
            usb = None
            pipewire_default = None
            for i, d in enumerate(devices):
                name = d["name"].lower()
                if d["max_input_channels"] <= 0:
                    continue
                if "hd-audio generic" in name and "analog" in name:
                    onboard = i
                elif "usb" in name:
                    usb = i
                elif name == "default" or name == "pipewire":
                    pipewire_default = i

            chosen = pipewire_default or onboard or usb
            if chosen is not None:
                d = devices[chosen]
                log.info("Using mic: [%d] %s (in=%d, default_sr=%.0f)",
                         chosen, d["name"], d["max_input_channels"],
                         d["default_samplerate"])
            else:
                log.warning("No mic device found, using system default")
            return chosen
        except Exception as e:
            log.warning("Could not enumerate audio devices: %s", e)
        return None

    @staticmethod
    def _downsample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Downsample audio by simple decimation (from_rate must be integer multiple of to_rate)."""
        if from_rate == to_rate:
            return audio
        ratio = from_rate // to_rate
        return audio[::ratio]

    def _highpass(self, audio: np.ndarray, sr: int, cutoff: int = 80) -> np.ndarray:
        """Second-order Butterworth high-pass filter to remove sub-bass rumble."""
        if self._hp_b is None:
            from scipy.signal import butter
            self._hp_b, self._hp_a = butter(2, cutoff, btype="high", fs=sr)
        from scipy.signal import lfilter
        return lfilter(self._hp_b, self._hp_a, audio)

    _hp_b = None
    _hp_a = None

    def _capture_loop(self):
        """Blocking audio capture loop (runs in dedicated thread)."""
        import sounddevice as sd

        self._load_vad()

        device_idx = self._find_mic()
        target_sr = config.SAMPLE_RATE

        # Detect native sample rate from device
        import sounddevice as sd
        if device_idx is not None:
            dev_info = sd.query_devices(device_idx)
            native_sr = int(dev_info["default_samplerate"])
        else:
            native_sr = NATIVE_RATE

        # PipeWire may report 44100 but resamples transparently.
        # Use 48000 for clean 3:1 decimation to 16kHz.
        if native_sr == 44100:
            native_sr = 48000
            log.info("Overriding native rate 44100->48000 (PipeWire resamples)")

        # Chunk size at native rate, so downsampled chunk matches config
        native_chunk = config.CHUNK_FRAMES * (native_sr // target_sr)

        # State
        recording = False
        audio_buffer = []  # stores 16kHz downsampled chunks
        silence_count = 0
        pre_speech_ring = collections.deque(maxlen=config.PRE_SPEECH_CHUNKS)
        last_transcription = ""  # for hybrid endpointing
        live_check_counter = 0

        def _put_segment(audio: np.ndarray):
            """Thread-safe enqueue to asyncio."""
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self.speech_queue.put_nowait, audio)

        log.info("Starting audio capture (device=%s, native_sr=%d, target_sr=%d, chunk=%d)",
                 device_idx, native_sr, target_sr, native_chunk)

        # Detect channel count — some onboard codecs only support stereo
        if device_idx is not None:
            dev_info = sd.query_devices(device_idx)
            in_channels = dev_info["max_input_channels"]
            # Try mono first; if device needs stereo, use 2 and take ch0
            channels = 1 if in_channels == 1 else 2
        else:
            channels = 1

        self.diag_device_name = str(device_idx)
        log.info("Opening stream: device=%s, sr=%d, ch=%d", device_idx, native_sr, channels)

        with sd.InputStream(
            samplerate=native_sr,
            channels=channels,
            dtype="float32",
            blocksize=native_chunk,
            device=device_idx,
        ) as stream:
            while self._running:
                data, overflowed = stream.read(native_chunk)
                if overflowed:
                    log.warning("Audio buffer overflow")
                    self.diag_overflows += 1
                # Take first channel only (mono)
                if channels > 1:
                    raw_audio = data[:, 0]
                else:
                    raw_audio = data.flatten()
                # Downsample to 16kHz for VAD and STT
                audio = self._downsample(raw_audio, native_sr, target_sr)
                # 80Hz high-pass: removes DC offset (0.014) and 60Hz mains hum (12% of noise)
                audio = self._highpass(audio, target_sr)

                # Run VAD on 512-sample windows
                vad_prob = 0.0
                window_size = 512
                for i in range(0, len(audio) - window_size + 1, window_size):
                    p = self._check_vad(audio[i : i + window_size])
                    vad_prob = max(vad_prob, p)

                is_speech = vad_prob >= config.VAD_THRESHOLD

                # Update diagnostics
                self.diag_chunks_processed += 1
                self.diag_last_peak = float(np.max(np.abs(audio)))
                self.diag_last_vad_prob = float(vad_prob)

                # Periodic diagnostic logging (every ~10s)
                if self.diag_chunks_processed % 40 == 0:
                    log.info("[DIAG] chunks=%d peak=%.4f vad=%.3f overflows=%d recording=%s suppressed=%s",
                             self.diag_chunks_processed, self.diag_last_peak,
                             self.diag_last_vad_prob, self.diag_overflows,
                             recording, self.suppressed)

                # Suppress: treat as silence during TTS playback OR when the
                # user has muted hearing from LT-OS. Mic + VAD keep running
                # (diagnostics stay fresh) but no segment is enqueued to STT.
                if self.suppressed or self.hearing_muted:
                    is_speech = False
                    if recording:
                        reason = "TTS suppression" if self.suppressed else "hearing muted"
                        log.debug("Discarding speech buffer (%s)", reason)
                        recording = False
                        audio_buffer = []
                        silence_count = 0
                        last_transcription = ''
                    continue

                if not recording:
                    pre_speech_ring.append(audio)
                    if is_speech:
                        recording = True
                        silence_count = 0
                        audio_buffer = list(pre_speech_ring) + [audio]
                        log.debug("Speech onset detected (prob=%.2f)", vad_prob)
                else:
                    audio_buffer.append(audio)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1

                    # Hybrid endpointing
                    complete_threshold = config.SILENCE_CHUNKS_COMPLETE
                    incomplete_threshold = config.SILENCE_CHUNKS_INCOMPLETE

                    if silence_count >= complete_threshold:
                        if looks_complete(last_transcription):
                            # Fast finalize — sentence is complete
                            segment = np.concatenate(audio_buffer)
                            _put_segment(segment)
                            log.info("Finalized speech (complete, %.1fs)",
                                     len(segment) / target_sr)
                            recording = False
                            audio_buffer = []
                            silence_count = 0
                            last_transcription = ""
                            continue

                    if silence_count >= incomplete_threshold:
                        # Timeout finalize — waited long enough
                        segment = np.concatenate(audio_buffer)
                        _put_segment(segment)
                        log.info("Finalized speech (timeout, %.1fs)",
                                 len(segment) / target_sr)
                        recording = False
                        audio_buffer = []
                        silence_count = 0
                        last_transcription = ""

                    # Periodic live transcription for endpointing decisions.
                    # Submit is non-blocking; we read whatever the worker has
                    # most recently transcribed (may lag by one network round-trip,
                    # which is acceptable for endpointing heuristics).
                    live_check_counter += 1
                    if recording and silence_count >= 1 and live_check_counter % 3 == 0:
                        if self._live_transcribe_fn and self._live_audio_queue is not None:
                            segment_so_far = np.concatenate(audio_buffer)
                            self._submit_live_audio(segment_so_far)
                            last_transcription = self._latest_live_text

    _live_transcribe_fn = None

    def set_live_transcribe(self, fn):
        """Set a callback for live transcription during recording.

        fn(audio_np) -> str, called from a dedicated worker thread (NOT the
        capture thread) so its HTTP round-trip never blocks capture/VAD.
        Must be thread-safe (typically uses requests/urllib, not async).
        """
        self._live_transcribe_fn = fn
        if self._live_worker_thread is None:
            # Capacity 1: we only ever care about the newest audio segment.
            self._live_audio_queue = queue.Queue(maxsize=1)
            self._live_worker_thread = threading.Thread(
                target=self._live_worker_loop, daemon=True
            )
            self._live_worker_thread.start()
            log.info("Live transcription worker thread started")

    def _submit_live_audio(self, audio: np.ndarray):
        """Non-blocking submit. Drops oldest pending audio so worker always
        operates on the freshest segment."""
        try:
            self._live_audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._live_audio_queue.put_nowait(audio)
        except queue.Full:
            pass

    def _live_worker_loop(self):
        """Worker loop: pull audio, run transcription, publish latest text."""
        while self._running or self._live_worker_thread is not None:
            try:
                audio = self._live_audio_queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running:
                    return
                continue
            if not self._live_transcribe_fn:
                continue
            try:
                text = self._live_transcribe_fn(audio)
                self._latest_live_text = text or ""
            except Exception as e:
                log.warning("live_transcribe failed: %s", e)

    def set_hearing(self, enabled: bool):
        """Toggle hearing. Whisper-server stays running; this only gates
        whether mic frames get enqueued for STT. Unmute is instant."""
        prev = self.hearing_muted
        self.hearing_muted = not bool(enabled)
        if prev != self.hearing_muted:
            log.info("Hearing %s", "muted" if self.hearing_muted else "unmuted")
            # Bundle D 2026-05-14: persist the change so a reboot doesn't
            # silently reset the toggle. Read-back at next AudioCapture init.
            try:
                from persistence import runtime_toggles as _toggles
                _toggles.set("hearing_enabled", not self.hearing_muted)
            except Exception as e:
                log.warning("hearing persist failed: %s", e)

    @property
    def is_hearing_enabled(self) -> bool:
        return not self.hearing_muted

    async def start(self, loop: asyncio.AbstractEventLoop):
        """Start the capture thread."""
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("Audio capture thread started")

    async def stop(self):
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
