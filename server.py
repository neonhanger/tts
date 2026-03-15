import asyncio
import base64
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
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

API_KEY = os.environ.get("API_KEY")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")
MAX_RETRIES = 2
SILENCE_PAD = 0.15

# ---------------------------------------------------------------------------
# Engine abstraction
# ---------------------------------------------------------------------------

class TTSEngine(ABC):
    """Base class for TTS engines."""

    @property
    @abstractmethod
    def engine_id(self) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @abstractmethod
    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        """Generate speech to file_wave."""
        ...


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


# Placeholder engines for future models

class CosyEngine(TTSEngine):
    """CosyVoice2 engine — low latency, emotional control, dialect support."""

    def __init__(self, device: str = "cuda"):
        self._model = None
        self._device = device

    @property
    def engine_id(self) -> str:
        return "cosy"

    @property
    def name(self) -> str:
        return "Cosy (expressive, low latency)"

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        # TODO: Install and load CosyVoice2-0.5B
        # from cosyvoice.cli.cosyvoice import AutoModel
        # self._model = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')
        print("[cosy] CosyVoice2 not yet installed — skipping")

    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        if not self.is_loaded():
            raise RuntimeError("CosyVoice2 engine not loaded")
        # TODO: Implement CosyVoice2 inference
        # import torchaudio
        # for i, j in enumerate(self._model.inference_zero_shot(gen_text, ref_text, ref_file)):
        #     torchaudio.save(file_wave, j['tts_speech'], self._model.sample_rate)
        #     break  # just take first result
        raise NotImplementedError("CosyVoice2 inference not yet implemented")


class HiggsEngine(TTSEngine):
    """Higgs Audio V2 engine — highest quality, expressive, needs quantization on 16GB."""

    def __init__(self, device: str = "cuda"):
        self._model = None
        self._device = device

    @property
    def engine_id(self) -> str:
        return "higgs"

    @property
    def name(self) -> str:
        return "Higgs (highest quality)"

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        # TODO: Install and load Higgs Audio V2 (requires quantization for 16GB VRAM)
        # from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        # self._model = HiggsAudioServeEngine(MODEL_PATH, TOKENIZER_PATH, device=self._device)
        print("[higgs] Higgs Audio V2 not yet installed — skipping")

    def infer(self, gen_text: str, ref_file: str, ref_text: str, file_wave: str):
        if not self.is_loaded():
            raise RuntimeError("Higgs Audio V2 engine not loaded")
        raise NotImplementedError("Higgs Audio V2 inference not yet implemented")


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

    # Register and load engines
    for EngineClass in [FlowEngine, CosyEngine, HiggsEngine]:
        engine = EngineClass(device=TTS_DEVICE)
        ENGINES[engine.engine_id] = engine
        try:
            engine.load()
        except Exception as e:
            print(f"[{engine.engine_id}] Failed to load: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _meta_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "meta.json"


def _audio_path(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "reference.wav"


def _refs_dir(voice_id: str) -> Path:
    return VOICES_DIR / voice_id / "refs"


def _get_best_ref(voice_id: str) -> str:
    refs = _refs_dir(voice_id)
    if refs.exists():
        wavs = list(refs.glob("*.wav"))
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


def _chunk_text(text: str, max_chars: int = 100) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
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
    return chunks or [text]


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
    }


@app.get("/engines")
def list_engines():
    """List available TTS engines."""
    return [
        {"id": eid, "name": e.name, "loaded": e.is_loaded()}
        for eid, e in ENGINES.items()
    ]


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
        # Try ffmpeg for OGG/Opus etc
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "24000", "-ac", "1", str(ref_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            shutil.rmtree(voice_dir)
            raise HTTPException(400, "Could not read audio file")

    if raw_path.exists() and raw_path != ref_path:
        raw_path.unlink()

    ref_text = _transcribe_reference(str(ref_path))
    if ref_text:
        _ref_text_path(voice_id).write_text(ref_text, encoding="utf-8")

    meta = {
        "voice_id": voice_id,
        "original_filename": file.filename,
        "ref_text": ref_text or "(auto-transcribe)",
    }
    _meta_path(voice_id).write_text(json.dumps(meta), encoding="utf-8")
    return meta


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
    result = []
    for c in clips:
        data, sr = sf.read(str(c))
        result.append({"name": c.name, "duration": round(len(data) / sr, 1)})
    return {"voice_id": voice_id, "clips": result, "mode": "multi-reference"}


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


# --- Speech generation ---

class SpeakRequest(BaseModel):
    text: str
    voice_id: str
    engine: Optional[str] = None


@app.post("/speak", dependencies=[Depends(check_api_key)])
async def speak(req: SpeakRequest):
    engine = _get_engine(req.engine)
    ref_audio = _get_best_ref(req.voice_id)
    ref_text = _get_ref_text(req.voice_id)
    _ = _load_meta(req.voice_id)

    if not Path(ref_audio).exists():
        raise HTTPException(404, "Reference audio missing for this voice")

    job_id = uuid.uuid4().hex[:8]
    with _queue_lock:
        _queue.append(job_id)

    try:
        async with _gpu_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, _generate_full, engine, req.text, ref_audio, ref_text
            )
    finally:
        with _queue_lock:
            if job_id in _queue:
                _queue.remove(job_id)

    return FileResponse(result, media_type="audio/mpeg", filename="speech.mp3")


def _generate_full(engine: TTSEngine, text: str, ref_audio: str, ref_text: str) -> str:
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


def _generate_chunk(engine: TTSEngine, chunk_text: str, ref_audio: str, ref_text: str, pad: bool) -> str:
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        _infer_with_retry(engine, chunk_text, ref_audio, ref_text, tmp_wav.name)
        if pad:
            _pad_silence(tmp_wav.name)
        return base64.b64encode(Path(tmp_wav.name).read_bytes()).decode("ascii")
    finally:
        if os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)


# Job-based streaming
_jobs: dict = {}
_jobs_lock = threading.Lock()


@app.post("/speak/start", dependencies=[Depends(check_api_key)])
async def speak_start(req: SpeakRequest):
    engine = _get_engine(req.engine)
    ref_audio = _get_best_ref(req.voice_id)
    ref_text = _get_ref_text(req.voice_id)
    _ = _load_meta(req.voice_id)

    if not Path(ref_audio).exists():
        raise HTTPException(404, "Reference audio missing for this voice")

    job_id = uuid.uuid4().hex[:8]
    chunks = _chunk_text(req.text)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "total": len(chunks),
            "current": 0,
            "chunks": [],
            "error": None,
        }

    async def run_job():
        async with _gpu_lock:
            loop = asyncio.get_event_loop()
            for i, chunk_text in enumerate(chunks):
                with _jobs_lock:
                    _jobs[job_id]["status"] = "generating"
                    _jobs[job_id]["current"] = i + 1
                try:
                    b64 = await loop.run_in_executor(
                        None, _generate_chunk, engine, chunk_text, ref_audio, ref_text, (i == 0)
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


@app.get("/queue")
def queue_status():
    with _queue_lock:
        return {"queue_size": len(_queue), "jobs": list(_queue)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
