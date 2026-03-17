import asyncio
import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Depends, Request,
    Query, WebSocket, WebSocketDisconnect, Form,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

API_KEY = os.environ.get("API_KEY")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")
MAX_RETRIES = 2
SILENCE_PAD = 0.15
CACHE_MAX_SIZE = int(os.environ.get("CACHE_MAX_SIZE", "100"))

# WhatsApp Cloud API (#2)
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_ID", "")
WHATSAPP_DEFAULT_VOICE = os.environ.get("WHATSAPP_DEFAULT_VOICE", "default")

# Post-processing defaults (#8, #16)
PP_NORMALIZE = os.environ.get("PP_NORMALIZE", "1") == "1"
PP_TARGET_LUFS = float(os.environ.get("PP_TARGET_LUFS", "-16"))
PP_HIGHPASS_HZ = int(os.environ.get("PP_HIGHPASS_HZ", "80"))

# ---------------------------------------------------------------------------
# Engine abstraction
# ---------------------------------------------------------------------------

class TTSEngine(ABC):
    @property
    @abstractmethod
    def engine_id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str): ...


class FlowEngine(TTSEngine):
    """F5-TTS flow matching engine — good all-rounder, fast voice cloning."""

    def __init__(self, device: str = "cuda"):
        self._model = None
        self._device = device

    @property
    def engine_id(self) -> str:
        return "flow"

    @property
    def name(self) -> str:
        return "Flow (fast, reliable)"

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        from f5_tts.api import F5TTS
        self._model = F5TTS(device=self._device)
        print(f"[flow] F5-TTS loaded on {self._device}")

    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        self._model.infer(
            gen_text=gen_text,
            ref_file=ref_file,
            ref_text=ref_text,
            file_wave=file_wave,
        )


class CosyEngine(TTSEngine):
    """CosyVoice2 — low latency, emotional control, dialect support.
    Runs in a separate venv via subprocess worker."""

    ENGINE_DIR = Path(__file__).parent / "engines" / "cosyvoice"
    WORKER_SCRIPT = ENGINE_DIR / "worker.py"
    PYTHON = ENGINE_DIR / "venv" / "Scripts" / "python.exe"

    def __init__(self, device: str = "cuda"):
        self._ready = False
        self._device = device

    @property
    def engine_id(self) -> str:
        return "cosy"

    @property
    def name(self) -> str:
        return "Cosy (expressive, low latency)"

    def is_loaded(self) -> bool:
        return self._ready

    def load(self):
        if not self.WORKER_SCRIPT.exists() or not self.PYTHON.exists():
            print("[cosy] CosyVoice2 engine not installed — skipping")
            return
        model_dir = self.ENGINE_DIR / "pretrained_models" / "CosyVoice2-0.5B"
        if not model_dir.exists():
            print("[cosy] CosyVoice2 model not downloaded — skipping")
            return
        self._ready = True
        print("[cosy] CosyVoice2 ready (subprocess worker)")

    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        if not self._ready:
            raise RuntimeError("CosyVoice2 engine not ready")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        result = subprocess.run(
            [
                str(self.PYTHON), str(self.WORKER_SCRIPT),
                "--text", gen_text,
                "--ref_audio", str(Path(ref_file).resolve()),
                "--ref_text", ref_text,
                "--output", str(Path(file_wave).resolve()),
            ],
            capture_output=True, text=True, timeout=300,
            cwd=str(self.ENGINE_DIR),
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CosyVoice2 worker failed: {result.stderr[-500:]}")
        if not os.path.exists(file_wave):
            raise RuntimeError("CosyVoice2 worker produced no output")


class HiggsEngine(TTSEngine):
    """Higgs Audio V2 — highest quality. Runs in a separate venv via subprocess worker."""

    ENGINE_DIR = Path(__file__).parent / "engines" / "higgs"
    WORKER_SCRIPT = ENGINE_DIR / "worker.py"
    PYTHON = ENGINE_DIR / "venv" / "Scripts" / "python.exe"

    def __init__(self, device: str = "cuda"):
        self._ready = False
        self._device = device

    @property
    def engine_id(self) -> str:
        return "higgs"

    @property
    def name(self) -> str:
        return "Higgs (highest quality)"

    def is_loaded(self) -> bool:
        return self._ready

    def load(self):
        if not self.WORKER_SCRIPT.exists() or not self.PYTHON.exists():
            print("[higgs] Higgs Audio V2 engine not installed — skipping")
            return
        model_dir = self.ENGINE_DIR / "models" / "generation"
        if not model_dir.exists():
            print("[higgs] Higgs Audio V2 model not downloaded — skipping")
            return
        self._ready = True
        print("[higgs] Higgs Audio V2 ready (subprocess worker)")

    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        if not self._ready:
            raise RuntimeError("Higgs Audio V2 engine not ready")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        result = subprocess.run(
            [
                str(self.PYTHON), str(self.WORKER_SCRIPT),
                "--text", gen_text,
                "--ref_audio", str(Path(ref_file).resolve()),
                "--ref_text", ref_text,
                "--output", str(Path(file_wave).resolve()),
            ],
            capture_output=True, text=True, timeout=600,
            cwd=str(self.ENGINE_DIR),
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Higgs Audio V2 worker failed: {result.stderr[-500:]}")
        if not os.path.exists(file_wave):
            raise RuntimeError("Higgs Audio V2 worker produced no output")


class EnsembleEngine(TTSEngine):
    """Model ensemble (#15) — runs multiple engines, picks best output by quality score."""

    def __init__(self, engines: dict):
        self._engines = engines

    @property
    def engine_id(self) -> str:
        return "ensemble"

    @property
    def name(self) -> str:
        return "Ensemble (best of multiple)"

    def is_loaded(self) -> bool:
        loaded = [e for e in self._engines.values() if e.engine_id != "ensemble" and e.is_loaded()]
        return len(loaded) >= 2

    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        loaded = [e for e in self._engines.values() if e.engine_id != "ensemble" and e.is_loaded()]
        if len(loaded) < 2:
            raise RuntimeError("Ensemble needs at least 2 loaded engines")
        best_score, best_path = -1, None
        for engine in loaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                engine.infer(gen_text, ref_file, ref_text, tmp.name)
                score = _score_audio_quality(tmp.name)
                if score > best_score:
                    if best_path:
                        os.unlink(best_path)
                    best_score = score
                    best_path = tmp.name
                else:
                    os.unlink(tmp.name)
            except Exception:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        if best_path:
            shutil.move(best_path, file_wave)
        else:
            raise RuntimeError("All engines failed in ensemble")


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------
ENGINES: dict[str, TTSEngine] = {}
DEFAULT_ENGINE = "flow"


def _get_engine(engine_id: Optional[str] = None) -> TTSEngine:
    eid = engine_id or DEFAULT_ENGINE
    engine = ENGINES.get(eid)
    if not engine:
        raise HTTPException(404, f"Engine '{eid}' not found. Available: {list(ENGINES.keys())}")
    if not engine.is_loaded():
        raise HTTPException(503, f"Engine '{eid}' is not loaded")
    return engine


# ---------------------------------------------------------------------------
# Response cache (#10)
# ---------------------------------------------------------------------------

class ResponseCache:
    def __init__(self, max_size: int, cache_dir: Path):
        self._max_size = max_size
        self._dir = cache_dir
        self._dir.mkdir(exist_ok=True)
        self._index: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._load_index()

    def _load_index(self):
        index_path = self._dir / "index.json"
        if index_path.exists():
            try:
                for e in json.loads(index_path.read_text()):
                    if (self._dir / e["file"]).exists():
                        self._index[e["key"]] = e["file"]
            except Exception:
                pass

    def _save_index(self):
        entries = [{"key": k, "file": v} for k, v in self._index.items()]
        (self._dir / "index.json").write_text(json.dumps(entries))

    def _key(self, text: str, voice_id: str, engine: str) -> str:
        return hashlib.sha256(f"{text}|{voice_id}|{engine}".encode()).hexdigest()[:16]

    def get(self, text: str, voice_id: str, engine: str) -> Optional[str]:
        key = self._key(text, voice_id, engine)
        with self._lock:
            if key in self._index:
                self._index.move_to_end(key)
                path = self._dir / self._index[key]
                if path.exists():
                    return str(path)
                del self._index[key]
        return None

    def put(self, text: str, voice_id: str, engine: str, audio_path: str):
        key = self._key(text, voice_id, engine)
        ext = Path(audio_path).suffix
        cached_file = f"{key}{ext}"
        shutil.copy2(audio_path, str(self._dir / cached_file))
        with self._lock:
            self._index[key] = cached_file
            self._index.move_to_end(key)
            while len(self._index) > self._max_size:
                _, old_file = self._index.popitem(last=False)
                old_path = self._dir / old_file
                if old_path.exists():
                    old_path.unlink()
            self._save_index()

    def clear(self):
        with self._lock:
            for f in self._dir.iterdir():
                if f.name != ".gitkeep":
                    f.unlink()
            self._index.clear()
            self._save_index()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._index)


_cache = ResponseCache(CACHE_MAX_SIZE, CACHE_DIR)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Voice Cloning TTS Service")

_gpu_lock = None
_queue: deque = deque()
_queue_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def check_api_key(request: Request):
    if API_KEY is None:
        return
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global _gpu_lock
    _gpu_lock = asyncio.Lock()

    for EngineClass in [FlowEngine, CosyEngine, HiggsEngine]:
        engine = EngineClass(device=TTS_DEVICE)
        ENGINES[engine.engine_id] = engine
        try:
            engine.load()
        except Exception as e:
            print(f"[{engine.engine_id}] Failed to load: {e}")

    # Register ensemble engine (#15)
    ensemble = EnsembleEngine(ENGINES)
    ENGINES["ensemble"] = ensemble
    if ensemble.is_loaded():
        print("[ensemble] Ensemble engine ready")
    else:
        print("[ensemble] Not enough loaded engines for ensemble")


# ---------------------------------------------------------------------------
# SSML parser (#5)
# ---------------------------------------------------------------------------

def _parse_ssml(text: str) -> tuple[str, list[dict]]:
    """Parse basic SSML tags. Returns clean text and break instructions."""
    breaks = []
    # Strip emphasis/prosody/phoneme tags but keep inner text
    clean = re.sub(r'</?(?:emphasis|prosody|say-as|phoneme|speak)[^>]*>', '', text)
    # Extract <break time="500ms"/> tags
    parts = re.split(r'<break\s+time=["\']([^"\']+)["\']\s*/?>', clean)
    clean_text = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            clean_text += part
        else:
            time_str = part.strip()
            if time_str.endswith("ms"):
                ms = int(time_str[:-2])
            elif time_str.endswith("s"):
                ms = int(float(time_str[:-1]) * 1000)
            else:
                ms = 500
            breaks.append({"char_pos": len(clean_text), "duration_ms": ms})
    # Strip any remaining tags
    clean_text = re.sub(r'<[^>]+/?>', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text, breaks


def _apply_breaks(wav_path: str, breaks: list[dict], text: str):
    """Insert silence at break positions in generated audio."""
    if not breaks:
        return
    data, sr = sf.read(wav_path)
    text_len = max(len(text), 1)
    segments = []
    last_sample = 0
    for brk in sorted(breaks, key=lambda b: b["char_pos"]):
        ratio = brk["char_pos"] / text_len
        split_sample = min(int(ratio * len(data)), len(data))
        segments.append(data[last_sample:split_sample])
        silence_samples = int(sr * brk["duration_ms"] / 1000)
        if data.ndim > 1:
            segments.append(np.zeros((silence_samples, data.shape[1]), dtype=data.dtype))
        else:
            segments.append(np.zeros(silence_samples, dtype=data.dtype))
        last_sample = split_sample
    segments.append(data[last_sample:])
    sf.write(wav_path, np.concatenate(segments), sr)


# ---------------------------------------------------------------------------
# Language detection (#6)
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> dict:
    """Detect language from text using Unicode script analysis."""
    clean = re.sub(r'\s+', '', text)
    if not clean:
        return {"language": "en", "confidence": 0.0, "script": "latin"}

    counts = {
        "cjk_zh": sum(1 for c in clean if '\u4e00' <= c <= '\u9fff'),
        "japanese": sum(1 for c in clean if '\u3040' <= c <= '\u30ff' or '\u31f0' <= c <= '\u31ff'),
        "hangul": sum(1 for c in clean if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff'),
        "cyrillic": sum(1 for c in clean if '\u0400' <= c <= '\u04ff'),
        "arabic": sum(1 for c in clean if '\u0600' <= c <= '\u06ff'),
        "devanagari": sum(1 for c in clean if '\u0900' <= c <= '\u097f'),
        "thai": sum(1 for c in clean if '\u0e00' <= c <= '\u0e7f'),
        "latin": sum(1 for c in clean if 'a' <= c.lower() <= 'z' or '\u00c0' <= c <= '\u024f'),
    }

    total = len(clean)
    best_script = max(counts, key=counts.get)
    best_count = counts[best_script]
    confidence = round(best_count / total, 2) if total > 0 else 0

    lang_map = {
        "cjk_zh": ("zh", "cjk"),
        "japanese": ("ja", "japanese"),
        "hangul": ("ko", "hangul"),
        "cyrillic": ("ru", "cyrillic"),
        "arabic": ("ar", "arabic"),
        "devanagari": ("hi", "devanagari"),
        "thai": ("th", "thai"),
        "latin": ("en", "latin"),
    }
    lang, script = lang_map.get(best_script, ("en", "latin"))
    return {"language": lang, "confidence": confidence, "script": script}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "meta.json"


def _audio_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "reference.wav"


def _refs_dir(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "refs"


def _emotions_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "emotions.json"


def _get_best_ref(voice_id: str, emotion: Optional[str] = None) -> str:
    """Pick a reference clip, optionally filtered by emotion (#13)."""
    refs = _refs_dir(voice_id)
    if refs.exists():
        wavs = list(refs.glob("*.wav"))
        if wavs and emotion:
            ep = _emotions_path(voice_id)
            if ep.exists():
                emotions = json.loads(ep.read_text())
                matching = [w for w in wavs if emotions.get(w.name) == emotion]
                if matching:
                    return str(random.choice(matching))
        if wavs:
            return str(random.choice(wavs))
    return str(_audio_path(voice_id))


def _ref_text_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "ref_text.txt"


def _load_meta(voice_id: str) -> dict:
    mp = _meta_path(voice_id)
    if not mp.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    return json.loads(mp.read_text(encoding="utf-8"))


def _get_ref_text(voice_id: str) -> str:
    rt = _ref_text_path(voice_id)
    if rt.exists():
        return rt.read_text(encoding="utf-8").strip()
    return ""


def _transcribe_reference(wav_path: str) -> str:
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(wav_path)
        return result.get("text", "").strip()
    except ImportError:
        try:
            from faster_whisper import WhisperModel
            wmodel = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = wmodel.transcribe(wav_path)
            return " ".join(s.text for s in segments).strip()
        except ImportError:
            return ""
    except Exception:
        return ""


def _pad_silence(wav_path: str, seconds: float = SILENCE_PAD):
    data, sr = sf.read(wav_path)
    silence = np.zeros(int(sr * seconds), dtype=data.dtype)
    if data.ndim > 1:
        silence = np.zeros((int(sr * seconds), data.shape[1]), dtype=data.dtype)
    sf.write(wav_path, np.concatenate([silence, data]), sr)


def _wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "128k"):
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, "-f", "mp3", mp3_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")


def _wav_to_ogg(wav_path: str, ogg_path: str):
    """Convert WAV to OGG/Opus for WhatsApp voice messages (#2)."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", ogg_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg ogg conversion failed: {result.stderr.decode()}")


# ---------------------------------------------------------------------------
# Text chunking (#3 — preserve quoted speech)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks, keeping quoted speech together."""
    text = text.strip()
    if not text:
        return [text] if text else []
    if len(text) <= max_chars:
        return [text]

    # Split into quoted and unquoted blocks
    blocks = []
    pattern = r'(\u201c[^\u201d]*\u201d|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'
    parts = re.split(pattern, text)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        is_quote = (
            (part.startswith('"') and part.endswith('"'))
            or (part.startswith("'") and part.endswith("'"))
            or (part.startswith('\u201c') and part.endswith('\u201d'))
        )
        blocks.append({"text": part, "quote": is_quote})

    # Merge short attribution ("she said") with preceding quote
    merged = []
    for block in blocks:
        if not block["quote"] and merged and merged[-1]["quote"]:
            stripped = block["text"].strip()
            if len(stripped) < 50 and re.match(
                r'^[,.]?\s*(?:he|she|they|it|I|we|[A-Z]\w+)\s+'
                r'(?:said|asked|replied|whispered|shouted|exclaimed|muttered|called|'
                r'cried|yelled|answered|demanded|insisted|suggested|added|continued|'
                r'began|explained|noted|murmured)',
                stripped,
            ):
                merged[-1]["text"] += " " + stripped
                continue
        merged.append(block)

    # Build chunks respecting quote boundaries
    chunks, current = [], ""
    for block in merged:
        block_text = block["text"]
        if block["quote"] and len(current) + len(block_text) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = ""
        if len(block_text) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_block(block_text, max_chars))
        elif len(current) + len(block_text) + 1 > max_chars:
            chunks.append(current.strip())
            current = block_text
        else:
            current = f"{current} {block_text}".strip() if current else block_text

    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _split_long_block(text: str, max_chars: int) -> list[str]:
    """Split a long block at sentence then clause boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for part in re.split(r'(?<=[,;:])\s+', sentence):
                if len(current) + len(part) + 1 > max_chars and current:
                    chunks.append(current.strip())
                    current = part
                else:
                    current = f"{current} {part}".strip() if current else part
        elif len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current.strip())
    return chunks


def _concat_wavs(wav_paths: list[str], output_path: str):
    all_data, sample_rate = [], None
    for wp in wav_paths:
        data, sr = sf.read(wp)
        if sample_rate is None:
            sample_rate = sr
        all_data.append(data)
    sf.write(output_path, np.concatenate(all_data), sample_rate)


def _split_audio_into_refs(wav_path: str, output_dir: str, min_dur: float = 5.0, max_dur: float = 10.0) -> list[str]:
    data, sr = sf.read(wav_path)
    if len(data) / sr <= max_dur:
        out = os.path.join(output_dir, "ref_001.wav")
        sf.write(out, data, sr)
        return [out]

    clips, pos, idx = [], 0, 1
    while pos < len(data):
        end = min(pos + int(8.0 * sr), len(data))
        if end < len(data):
            search_start = max(pos + int(min_dur * sr), end - int(sr))
            segment = np.abs(data[search_start:min(end + int(sr), len(data))])
            window = int(0.1 * sr)
            if len(segment) > window:
                rolling = np.convolve(segment, np.ones(window) / window, mode='valid')
                end = search_start + np.argmin(rolling) + window // 2
        clip = data[pos:end]
        if len(clip) / sr >= min_dur:
            out = os.path.join(output_dir, f"ref_{idx:03d}.wav")
            sf.write(out, clip, sr)
            clips.append(out)
            idx += 1
        pos = end
    return clips


# ---------------------------------------------------------------------------
# Voice quality scoring (#4)
# ---------------------------------------------------------------------------

def _score_voice_quality(wav_path: str) -> dict:
    """Analyze reference audio quality and return score + feedback."""
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    duration = len(data) / sr
    rms = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))
    clipping_ratio = float(np.sum(np.abs(data) > 0.99) / len(data))
    silence_threshold = 0.01
    silence_ratio = float(np.sum(np.abs(data) < silence_threshold) / len(data))

    score = 100
    issues = []
    tips = []

    if duration < 3:
        score -= 30
        issues.append("Too short (< 3s)")
        tips.append("Record at least 5 seconds of clear speech")
    elif duration < 5:
        score -= 10
        issues.append("Short (< 5s)")
        tips.append("5-10 seconds is ideal")
    elif duration > 15:
        score -= 10
        issues.append("Long (> 15s) — consider multi-reference mode")

    if clipping_ratio > 0.01:
        score -= 25
        issues.append("Audio clipping detected")
        tips.append("Reduce microphone input level")
    elif clipping_ratio > 0.001:
        score -= 10
        issues.append("Minor clipping")

    if silence_ratio > 0.5:
        score -= 20
        issues.append("Too much silence")
        tips.append("Trim silence from the clip")

    if rms < 0.01:
        score -= 20
        issues.append("Audio too quiet")
        tips.append("Speak closer to the microphone or boost volume")
    elif rms < 0.03:
        score -= 10
        issues.append("Audio is quiet")

    if rms > 0.4:
        score -= 10
        issues.append("Audio may be too loud / distorted")

    score = max(0, min(100, score))
    grade = "excellent" if score >= 80 else "good" if score >= 60 else "fair" if score >= 40 else "poor"

    return {
        "duration": round(duration, 1),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 2),
        "score": score,
        "grade": grade,
        "issues": issues,
        "tips": tips,
        "usable": score >= 40,
    }


def _score_audio_quality(wav_path: str) -> float:
    """Score generated audio quality (0-100) for cherry-pick / ensemble."""
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    rms = float(np.sqrt(np.mean(data ** 2)))
    peak = float(np.max(np.abs(data)))
    silence_ratio = float(np.sum(np.abs(data) < 0.005) / max(len(data), 1))
    clipping = float(np.sum(np.abs(data) > 0.99) / max(len(data), 1))

    score = 70.0
    score += min(1.0, rms / 0.05) * 15
    score -= silence_ratio * 20
    score -= clipping * 50
    score -= abs(peak - 0.7) * 10
    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Audio post-processing (#8, #16)
# ---------------------------------------------------------------------------

def _postprocess_audio(wav_path: str, normalize: bool = PP_NORMALIZE, highpass_hz: int = PP_HIGHPASS_HZ):
    """Apply post-processing: high-pass filter + loudness normalization."""
    filters = []
    if highpass_hz > 0:
        filters.append(f"highpass=f={highpass_hz}")
    if normalize:
        filters.append(f"loudnorm=I={PP_TARGET_LUFS}:TP=-1.5:LRA=11")
    if not filters:
        return

    tmp = wav_path + ".pp.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-af", ",".join(filters), tmp],
        capture_output=True,
    )
    if result.returncode == 0 and os.path.exists(tmp):
        shutil.move(tmp, wav_path)
    elif os.path.exists(tmp):
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_with_retry(engine: TTSEngine, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            engine.infer(gen_text, ref_file, ref_text, file_wave)
            return
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(0.5)
    raise last_err


def _generate_full(engine: TTSEngine, text: str, ref_audio: str, ref_text: str,
                   postprocess: bool = True, ssml_breaks: list = None) -> str:
    """Generate full audio, return path to MP3."""
    chunks = _chunk_text(text)
    chunk_wavs = []
    tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_mp3.close()
    combined_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    combined_wav.close()

    try:
        for chunk in chunks:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            chunk_wavs.append(tmp_wav.name)
            _infer_with_retry(engine, chunk, ref_audio, ref_text, tmp_wav.name)

        if len(chunk_wavs) == 1:
            combined_wav_path = chunk_wavs[0]
        else:
            _concat_wavs(chunk_wavs, combined_wav.name)
            combined_wav_path = combined_wav.name

        _pad_silence(combined_wav_path)

        if ssml_breaks:
            _apply_breaks(combined_wav_path, ssml_breaks, text)

        if postprocess:
            _postprocess_audio(combined_wav_path)

        _wav_to_mp3(combined_wav_path, tmp_mp3.name)
        return tmp_mp3.name

    except Exception as e:
        if os.path.exists(tmp_mp3.name):
            os.unlink(tmp_mp3.name)
        raise e
    finally:
        for wp in chunk_wavs:
            if os.path.exists(wp):
                os.unlink(wp)
        if os.path.exists(combined_wav.name) and combined_wav.name not in chunk_wavs:
            os.unlink(combined_wav.name)


def _generate_full_wav(engine: TTSEngine, text: str, ref_audio: str, ref_text: str,
                       postprocess: bool = True, ssml_breaks: list = None) -> str:
    """Generate full audio, return path to WAV (for cherry-pick scoring / WhatsApp)."""
    chunks = _chunk_text(text)
    chunk_wavs = []
    output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_wav.close()

    try:
        for chunk in chunks:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            chunk_wavs.append(tmp_wav.name)
            _infer_with_retry(engine, chunk, ref_audio, ref_text, tmp_wav.name)

        if len(chunk_wavs) == 1:
            shutil.copy2(chunk_wavs[0], output_wav.name)
        else:
            _concat_wavs(chunk_wavs, output_wav.name)

        _pad_silence(output_wav.name)

        if ssml_breaks:
            _apply_breaks(output_wav.name, ssml_breaks, text)

        if postprocess:
            _postprocess_audio(output_wav.name)

        return output_wav.name

    except Exception:
        if os.path.exists(output_wav.name):
            os.unlink(output_wav.name)
        raise
    finally:
        for wp in chunk_wavs:
            if os.path.exists(wp):
                os.unlink(wp)


def _generate_chunk(engine: TTSEngine, chunk_text: str, ref_audio: str, ref_text: str,
                    pad: bool, postprocess: bool = True) -> str:
    """Generate a chunk, return base64-encoded WAV (for poll-based streaming)."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        _infer_with_retry(engine, chunk_text, ref_audio, ref_text, tmp_wav.name)
        if pad:
            _pad_silence(tmp_wav.name)
        if postprocess:
            _postprocess_audio(tmp_wav.name)
        return base64.b64encode(Path(tmp_wav.name).read_bytes()).decode("ascii")
    finally:
        if os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)


def _generate_chunk_bytes(engine: TTSEngine, chunk_text: str, ref_audio: str, ref_text: str,
                          postprocess: bool = True) -> bytes:
    """Generate a chunk, return raw WAV bytes (for WebSocket streaming #7)."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        _infer_with_retry(engine, chunk_text, ref_audio, ref_text, tmp_wav.name)
        if postprocess:
            _postprocess_audio(tmp_wav.name)
        return Path(tmp_wav.name).read_bytes()
    finally:
        if os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SpeakRequest(BaseModel):
    text: str
    voice_id: str
    engine: Optional[str] = None
    emotion: Optional[str] = None    # #13 emotion control
    ssml: bool = False                # #5 SSML parsing
    postprocess: bool = True          # #8, #16 post-processing
    cache: bool = True                # #10 response caching


class MixRequest(BaseModel):
    voice_a: str
    voice_b: str
    ratio: float = 0.5


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return Path("index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    with _queue_lock:
        queue_size = len(_queue)
    return {
        "status": "ok",
        "engines": {eid: {"name": e.name, "loaded": e.is_loaded()} for eid, e in ENGINES.items()},
        "queue_size": queue_size,
        "cache_size": _cache.size,
    }


@app.get("/engines")
def list_engines():
    return [
        {"id": eid, "name": e.name, "loaded": e.is_loaded()}
        for eid, e in ENGINES.items()
    ]


# --- Language detection (#6) ---

@app.post("/detect-language")
def detect_language(text: str = Form(...)):
    return _detect_language(text)


# --- Voice management ---

@app.post("/voices", dependencies=[Depends(check_api_key)])
async def create_voice(file: UploadFile = File(...)):
    if file.content_type not in (
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3", "audio/ogg", "application/octet-stream",
    ):
        raise HTTPException(400, "Upload a WAV, MP3, or OGG file")

    voice_id = uuid.uuid4().hex[:12]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True)

    raw_path = voice_dir / f"upload_{file.filename}"
    contents = await file.read()
    raw_path.write_bytes(contents)

    ref_path = _audio_path(voice_id)
    try:
        data, sr = sf.read(str(raw_path))
        sf.write(str(ref_path), data, sr)
    except Exception:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "24000", "-ac", "1", str(ref_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            shutil.rmtree(voice_dir)
            raise HTTPException(400, "Could not read audio file")

    if raw_path.exists() and raw_path != ref_path:
        raw_path.unlink()

    # Quality scoring (#4)
    quality = _score_voice_quality(str(ref_path))

    ref_text = _transcribe_reference(str(ref_path))
    if ref_text:
        _ref_text_path(voice_id).write_text(ref_text, encoding="utf-8")

    meta = {
        "voice_id": voice_id,
        "original_filename": file.filename,
        "ref_text": ref_text or "(auto-transcribe)",
        "quality": quality,
    }
    _meta_path(voice_id).write_text(json.dumps(meta), encoding="utf-8")
    return meta


@app.get("/voices/{voice_id}/quality", dependencies=[Depends(check_api_key)])
def voice_quality(voice_id: str):
    """Get quality score for a voice's reference audio (#4)."""
    ref = _audio_path(voice_id)
    if not ref.exists():
        raise HTTPException(404, "Voice not found")
    return _score_voice_quality(str(ref))


@app.post("/voices/{voice_id}/refs", dependencies=[Depends(check_api_key)])
async def add_references(voice_id: str, file: UploadFile = File(...)):
    vdir = VOICES_DIR / voice_id
    if not vdir.exists():
        raise HTTPException(404, "Voice not found")

    raw_path = vdir / f"upload_long_{file.filename}"
    contents = await file.read()
    raw_path.write_bytes(contents)

    wav_path = vdir / "upload_long.wav"
    try:
        data, sr = sf.read(str(raw_path))
        sf.write(str(wav_path), data, sr)
    except Exception:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "24000", "-ac", "1", str(wav_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raw_path.unlink()
            raise HTTPException(400, "Could not read audio file")

    if raw_path.exists():
        raw_path.unlink()

    refs = _refs_dir(voice_id)
    if refs.exists():
        shutil.rmtree(refs)
    refs.mkdir(parents=True)

    clips = _split_audio_into_refs(str(wav_path), str(refs))
    wav_path.unlink()

    meta = _load_meta(voice_id)
    meta["ref_count"] = len(clips)
    meta["mode"] = "multi-reference"
    _meta_path(voice_id).write_text(json.dumps(meta), encoding="utf-8")

    return {"voice_id": voice_id, "ref_clips": len(clips), "mode": "multi-reference"}


@app.get("/voices/{voice_id}/refs", dependencies=[Depends(check_api_key)])
def list_references(voice_id: str):
    refs = _refs_dir(voice_id)
    if not refs.exists():
        return {"voice_id": voice_id, "clips": [], "mode": "single"}
    clips = sorted(refs.glob("*.wav"))
    ep = _emotions_path(voice_id)
    emotions = json.loads(ep.read_text()) if ep.exists() else {}
    result = []
    for c in clips:
        data, sr = sf.read(str(c))
        result.append({
            "name": c.name,
            "duration": round(len(data) / sr, 1),
            "emotion": emotions.get(c.name),
        })
    return {"voice_id": voice_id, "clips": result, "mode": "multi-reference"}


# --- Emotion tagging (#13) ---

@app.post("/voices/{voice_id}/refs/{ref_name}/emotion", dependencies=[Depends(check_api_key)])
def tag_emotion(voice_id: str, ref_name: str, emotion: str = Query(...)):
    """Tag a reference clip with an emotion label."""
    refs = _refs_dir(voice_id)
    if not refs.exists() or not (refs / ref_name).exists():
        raise HTTPException(404, "Reference clip not found")

    ep = _emotions_path(voice_id)
    emotions = json.loads(ep.read_text()) if ep.exists() else {}
    emotions[ref_name] = emotion
    ep.write_text(json.dumps(emotions), encoding="utf-8")
    return {"voice_id": voice_id, "ref": ref_name, "emotion": emotion}


@app.delete("/voices/{voice_id}/refs/{ref_name}/emotion", dependencies=[Depends(check_api_key)])
def untag_emotion(voice_id: str, ref_name: str):
    """Remove emotion tag from a reference clip."""
    ep = _emotions_path(voice_id)
    if ep.exists():
        emotions = json.loads(ep.read_text())
        emotions.pop(ref_name, None)
        ep.write_text(json.dumps(emotions), encoding="utf-8")
    return {"voice_id": voice_id, "ref": ref_name, "emotion": None}


# --- Voice mixing (#9) ---

@app.post("/voices/mix", dependencies=[Depends(check_api_key)])
def mix_voices(req: MixRequest):
    """Blend two voice profiles into a new voice."""
    ref_a = _audio_path(req.voice_a)
    ref_b = _audio_path(req.voice_b)
    if not ref_a.exists():
        raise HTTPException(404, f"Voice '{req.voice_a}' not found")
    if not ref_b.exists():
        raise HTTPException(404, f"Voice '{req.voice_b}' not found")

    data_a, sr_a = sf.read(str(ref_a))
    data_b, sr_b = sf.read(str(ref_b))

    # Resample if needed
    if sr_a != sr_b:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ref_b), "-ar", str(sr_a), "-ac", "1", tmp.name],
            capture_output=True,
        )
        data_b, _ = sf.read(tmp.name)
        os.unlink(tmp.name)

    if data_a.ndim > 1:
        data_a = np.mean(data_a, axis=1)
    if data_b.ndim > 1:
        data_b = np.mean(data_b, axis=1)

    min_len = min(len(data_a), len(data_b))
    ratio = max(0.0, min(1.0, req.ratio))
    blended = data_a[:min_len] * ratio + data_b[:min_len] * (1 - ratio)

    voice_id = uuid.uuid4().hex[:12]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(parents=True)

    ref_path = _audio_path(voice_id)
    sf.write(str(ref_path), blended, sr_a)

    ref_text = _transcribe_reference(str(ref_path))
    if ref_text:
        _ref_text_path(voice_id).write_text(ref_text, encoding="utf-8")

    meta = {
        "voice_id": voice_id,
        "original_filename": f"mix({req.voice_a}, {req.voice_b}, {ratio})",
        "ref_text": ref_text or "",
        "mode": "mixed",
        "source_voices": [req.voice_a, req.voice_b],
        "blend_ratio": ratio,
    }
    _meta_path(voice_id).write_text(json.dumps(meta), encoding="utf-8")
    return meta


@app.get("/voices", dependencies=[Depends(check_api_key)])
def list_voices():
    voices = []
    if VOICES_DIR.exists():
        for d in sorted(VOICES_DIR.iterdir()):
            mp = d / "meta.json"
            if mp.exists():
                voices.append(json.loads(mp.read_text(encoding="utf-8")))
    return voices


@app.get("/voices/{voice_id}/preview")
def preview_voice(voice_id: str):
    ref = _audio_path(voice_id)
    if not ref.exists():
        raise HTTPException(404, "Voice not found")
    return FileResponse(str(ref), media_type="audio/wav")


@app.delete("/voices/{voice_id}", dependencies=[Depends(check_api_key)])
def delete_voice(voice_id: str):
    if voice_id == "default":
        raise HTTPException(403, "Cannot delete the default voice")
    vdir = VOICES_DIR / voice_id
    if not vdir.exists():
        raise HTTPException(404, "Voice not found")
    shutil.rmtree(vdir)
    return {"deleted": voice_id}


# --- Fine-tuning scaffold (#11) ---

@app.post("/voices/{voice_id}/train", dependencies=[Depends(check_api_key)])
async def start_training(voice_id: str):
    """Start voice fine-tuning. Currently scaffold — use multi-reference mode."""
    _ = _load_meta(voice_id)
    raise HTTPException(
        501,
        "Voice fine-tuning not yet implemented. "
        "Upload longer audio via multi-reference mode for improved quality.",
    )


@app.get("/voices/{voice_id}/train/status", dependencies=[Depends(check_api_key)])
def training_status(voice_id: str):
    """Check fine-tuning status."""
    checkpoint_dir = VOICES_DIR / voice_id / "checkpoint"
    return {
        "voice_id": voice_id,
        "status": "complete" if checkpoint_dir.exists() else "not_started",
        "checkpoint": checkpoint_dir.exists(),
    }


# --- Speech generation ---

@app.post("/speak", dependencies=[Depends(check_api_key)])
async def speak(req: SpeakRequest):
    text = req.text
    ssml_breaks = []
    if req.ssml:
        text, ssml_breaks = _parse_ssml(text)

    lang = _detect_language(text)
    engine = _get_engine(req.engine)
    ref_audio = _get_best_ref(req.voice_id, req.emotion)
    ref_text = _get_ref_text(req.voice_id)
    _ = _load_meta(req.voice_id)

    if not Path(ref_audio).exists():
        raise HTTPException(404, "Reference audio missing for this voice")

    # Cache check (#10)
    engine_id = req.engine or DEFAULT_ENGINE
    if req.cache:
        cached = _cache.get(text, req.voice_id, engine_id)
        if cached:
            return FileResponse(
                cached, media_type="audio/mpeg", filename="speech.mp3",
                headers={"X-Cache": "hit", "X-Language": lang["language"]},
            )

    job_id = uuid.uuid4().hex[:8]
    with _queue_lock:
        _queue.append(job_id)

    try:
        async with _gpu_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _generate_full, engine, text, ref_audio, ref_text,
                req.postprocess, ssml_breaks,
            )
    finally:
        with _queue_lock:
            if job_id in _queue:
                _queue.remove(job_id)

    if req.cache:
        _cache.put(text, req.voice_id, engine_id, result)

    return FileResponse(
        result, media_type="audio/mpeg", filename="speech.mp3",
        headers={"X-Cache": "miss", "X-Language": lang["language"]},
    )


# --- Cherry-pick (#14) ---

@app.post("/speak/candidates", dependencies=[Depends(check_api_key)])
async def speak_candidates(req: SpeakRequest, takes: int = Query(default=3, ge=2, le=5)):
    """Generate multiple takes and return the best with quality scores."""
    text = req.text
    ssml_breaks = []
    if req.ssml:
        text, ssml_breaks = _parse_ssml(text)

    engine = _get_engine(req.engine)
    ref_audio = _get_best_ref(req.voice_id, req.emotion)
    ref_text = _get_ref_text(req.voice_id)
    _ = _load_meta(req.voice_id)

    if not Path(ref_audio).exists():
        raise HTTPException(404, "Reference audio missing")

    candidates = []
    async with _gpu_lock:
        loop = asyncio.get_event_loop()
        for i in range(takes):
            wav_path = await loop.run_in_executor(
                None, _generate_full_wav, engine, text, ref_audio, ref_text,
                req.postprocess, ssml_breaks,
            )
            score = _score_audio_quality(wav_path)
            candidates.append({"wav_path": wav_path, "score": score, "take": i + 1})

    candidates.sort(key=lambda c: c["score"], reverse=True)

    best = candidates[0]
    mp3_path = best["wav_path"].replace(".wav", ".mp3")
    _wav_to_mp3(best["wav_path"], mp3_path)

    scores = [{"take": c["take"], "score": round(c["score"], 1)} for c in candidates]

    for c in candidates:
        if os.path.exists(c["wav_path"]):
            os.unlink(c["wav_path"])

    return FileResponse(
        mp3_path, media_type="audio/mpeg", filename="speech_best.mp3",
        headers={
            "X-Candidates": json.dumps(scores),
            "X-Best-Score": str(round(best["score"], 1)),
        },
    )


# --- Job-based streaming ---

_jobs: dict = {}
_jobs_lock = threading.Lock()


@app.post("/speak/start", dependencies=[Depends(check_api_key)])
async def speak_start(req: SpeakRequest):
    text = req.text
    if req.ssml:
        text, _ = _parse_ssml(text)

    engine = _get_engine(req.engine)
    ref_audio = _get_best_ref(req.voice_id, req.emotion)
    ref_text = _get_ref_text(req.voice_id)
    _ = _load_meta(req.voice_id)

    if not Path(ref_audio).exists():
        raise HTTPException(404, "Reference audio missing for this voice")

    job_id = uuid.uuid4().hex[:8]
    chunks = _chunk_text(text)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "total": len(chunks),
            "current": 0,
            "chunks": [],
            "error": None,
        }

    postprocess = req.postprocess

    async def run_job():
        async with _gpu_lock:
            loop = asyncio.get_event_loop()
            for i, chunk_text in enumerate(chunks):
                with _jobs_lock:
                    _jobs[job_id]["status"] = "generating"
                    _jobs[job_id]["current"] = i + 1
                try:
                    b64 = await loop.run_in_executor(
                        None, _generate_chunk, engine, chunk_text, ref_audio, ref_text,
                        (i == 0), postprocess,
                    )
                    with _jobs_lock:
                        _jobs[job_id]["chunks"].append(b64)
                except Exception as e:
                    with _jobs_lock:
                        _jobs[job_id]["status"] = "error"
                        _jobs[job_id]["error"] = str(e)
                    return

            with _jobs_lock:
                _jobs[job_id]["status"] = "done"

    asyncio.create_task(run_job())
    return {"job_id": job_id, "total_chunks": len(chunks)}


@app.get("/speak/poll/{job_id}")
def speak_poll(job_id: str, after: int = 0):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "total": job["total"],
        "current": job["current"],
        "ready": len(job["chunks"]),
        "chunks": job["chunks"][after:],
        "error": job["error"],
    }


# --- WebSocket streaming (#7) ---

@app.websocket("/ws/speak")
async def ws_speak(ws: WebSocket):
    """Real-time streaming via WebSocket. Send JSON config, receive binary WAV chunks."""
    await ws.accept()
    try:
        data = await ws.receive_json()
        text = data.get("text", "")
        voice_id = data.get("voice_id", "")
        engine_id = data.get("engine")
        emotion = data.get("emotion")
        ssml = data.get("ssml", False)
        postprocess = data.get("postprocess", True)

        if ssml:
            text, _ = _parse_ssml(text)

        if not text or not voice_id:
            await ws.send_json({"type": "error", "detail": "text and voice_id required"})
            await ws.close()
            return

        engine = _get_engine(engine_id)
        ref_audio = _get_best_ref(voice_id, emotion)
        ref_text = _get_ref_text(voice_id)
        _ = _load_meta(voice_id)

        if not Path(ref_audio).exists():
            await ws.send_json({"type": "error", "detail": "Reference audio missing"})
            await ws.close()
            return

        chunks = _chunk_text(text)
        await ws.send_json({"type": "info", "total_chunks": len(chunks)})

        async with _gpu_lock:
            loop = asyncio.get_event_loop()
            for i, chunk_text in enumerate(chunks):
                await ws.send_json({
                    "type": "progress", "chunk": i + 1,
                    "total": len(chunks), "text": chunk_text,
                })
                wav_bytes = await loop.run_in_executor(
                    None, _generate_chunk_bytes, engine, chunk_text,
                    ref_audio, ref_text, postprocess,
                )
                await ws.send_bytes(wav_bytes)

        await ws.send_json({"type": "done"})
        await ws.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
            await ws.close()
        except Exception:
            pass


# --- Cache management (#10) ---

@app.get("/cache/stats")
def cache_stats():
    return {"size": _cache.size, "max_size": CACHE_MAX_SIZE}


@app.delete("/cache", dependencies=[Depends(check_api_key)])
def clear_cache():
    _cache.clear()
    return {"cleared": True}


# --- Queue ---

@app.get("/queue")
def queue_status():
    with _queue_lock:
        return {"queue_size": len(_queue), "jobs": list(_queue)}


# --- WhatsApp webhook (#2) ---

@app.get("/whatsapp/webhook")
def whatsapp_verify(request: Request):
    """WhatsApp webhook verification (GET)."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN and WHATSAPP_VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(403, "Verification failed")


@app.post("/whatsapp/webhook")
async def whatsapp_incoming(request: Request):
    """Handle incoming WhatsApp messages — generate TTS and reply with voice message."""
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_ID:
        raise HTTPException(
            501,
            "WhatsApp not configured. Set WHATSAPP_VERIFY_TOKEN, "
            "WHATSAPP_ACCESS_TOKEN, and WHATSAPP_PHONE_ID env vars.",
        )

    body = await request.json()

    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        message = value["messages"][0]
        from_number = message["from"]
        text = message.get("text", {}).get("body", "")
    except (KeyError, IndexError):
        return {"status": "no_message"}

    if not text:
        return {"status": "no_text"}

    # Voice command: !voice <voice_id> <text>
    voice_id = WHATSAPP_DEFAULT_VOICE
    if text.startswith("!voice "):
        parts = text.split(" ", 2)
        if len(parts) >= 3:
            voice_id = parts[1]
            text = parts[2]
        else:
            return {"status": "invalid_command"}

    engine = _get_engine()
    ref_audio = _get_best_ref(voice_id)
    ref_text = _get_ref_text(voice_id)
    try:
        _ = _load_meta(voice_id)
    except HTTPException:
        voice_id = WHATSAPP_DEFAULT_VOICE
        ref_audio = _get_best_ref(voice_id)
        ref_text = _get_ref_text(voice_id)
        _ = _load_meta(voice_id)

    # Generate WAV then convert to OGG/Opus for WhatsApp
    async with _gpu_lock:
        loop = asyncio.get_event_loop()
        wav_path = await loop.run_in_executor(
            None, _generate_full_wav, engine, text, ref_audio, ref_text, True, [],
        )

    ogg_path = wav_path.replace(".wav", ".ogg")
    _wav_to_ogg(wav_path, ogg_path)

    # TODO: Upload OGG to WhatsApp Media API and send voice message
    # 1. POST to https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/media
    #    with file=@{ogg_path}, type=audio/ogg, messaging_product=whatsapp
    # 2. POST to https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages
    #    with to={from_number}, type=audio, audio.id={media_id}
    # Requires: httpx or aiohttp for async HTTP calls

    for p in [wav_path, ogg_path]:
        if os.path.exists(p):
            os.unlink(p)

    return {
        "status": "processed",
        "from": from_number,
        "voice_id": voice_id,
        "text_length": len(text),
        "note": "WhatsApp media upload pending — implement Media API calls",
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
