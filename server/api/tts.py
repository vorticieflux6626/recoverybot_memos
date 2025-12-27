"""
memOS TTS API - Text-to-Speech endpoint using edge-tts
Provides high-quality Microsoft Edge voices for Recovery Bot

IMPORTANT: Returns WAV format for Android AudioTrack compatibility.
Edge-tts produces MP3 which must be converted to PCM WAV.
"""

import asyncio
import logging
import hashlib
import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Try to import edge_tts
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    edge_tts = None
    EDGE_TTS_AVAILABLE = False

# Try to import pydub for MP3 to WAV conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False

logger = logging.getLogger("memOS.tts")

router = APIRouter(prefix="/tts", tags=["TTS"])


def convert_mp3_to_pcm(mp3_data: bytes, sample_rate: int = 22050) -> bytes:
    """
    Convert MP3 audio data to raw PCM for Android AudioTrack compatibility.

    AudioTrack expects raw PCM data (no headers), so we need to convert the MP3
    format that edge-tts produces into raw 16-bit PCM samples.

    Args:
        mp3_data: Raw MP3 bytes from edge-tts
        sample_rate: Target sample rate (default 22050 for speech)

    Returns:
        Raw PCM bytes (16-bit, mono) that can be played directly by Android AudioTrack
    """
    if PYDUB_AVAILABLE:
        try:
            # Use pydub for conversion
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))

            # Convert to mono, 16-bit, target sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(sample_rate)
            audio = audio.set_sample_width(2)  # 16-bit = 2 bytes

            # Get raw PCM samples (no WAV header)
            pcm_data = audio.raw_data

            logger.debug(f"Converted MP3 ({len(mp3_data)} bytes) to raw PCM ({len(pcm_data)} bytes)")
            return pcm_data

        except Exception as e:
            logger.error(f"pydub conversion failed: {e}, trying ffmpeg")
            # Fall through to ffmpeg method

    # Fallback to ffmpeg subprocess - output raw PCM
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
            mp3_file.write(mp3_data)
            mp3_path = mp3_file.name

        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as pcm_file:
            pcm_path = pcm_file.name

        try:
            # Convert using ffmpeg to raw PCM (no container format)
            cmd = [
                'ffmpeg', '-y',
                '-i', mp3_path,
                '-f', 's16le',  # Raw signed 16-bit little-endian
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', str(sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                pcm_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {result.stderr.decode()}")
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

            with open(pcm_path, 'rb') as f:
                pcm_data = f.read()

            logger.debug(f"ffmpeg converted MP3 ({len(mp3_data)} bytes) to raw PCM ({len(pcm_data)} bytes)")
            return pcm_data

        finally:
            # Cleanup temp files
            try:
                os.unlink(mp3_path)
                os.unlink(pcm_path)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"MP3 to PCM conversion failed: {e}")
        # Return original MP3 as last resort (will cause garbled audio but not crash)
        return mp3_data

# TTS Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24

# Default voices for different personalities - alluring female voices
DEFAULT_VOICES = {
    "therapist": "en-US-AriaNeural",      # Warm, empathetic, intimately close
    "meditation": "en-US-AriaNeural",     # Calm, soothing female
    "coach": "en-US-JennyNeural",         # Encouraging, friendly
    "companion": "en-AU-NatashaNeural",   # Relaxed Australian warmth (default)
    "crisis": "en-US-SaraNeural",         # Soft, gentle, reassuring
    "sleep": "en-US-AriaNeural",          # Soothing, relaxing
    "seductive": "en-AU-NatashaNeural",   # Alluring, intimate, flirty
    "intimate": "en-US-AvaNeural",        # Warm, smooth, breathy
    "flirty": "en-GB-SoniaNeural",        # Sophisticated British allure
    "default": "en-AU-NatashaNeural"      # Relaxed Australian warmth
}

# Seductive/intimate voice settings - based on research for alluring TTS
# Lower pitch (-10 to -15 Hz) adds depth and allure
# Slower rate (0.80-0.85) for deliberate, lingering delivery
SEDUCTIVE_SETTINGS = {
    # Maximum seductive settings
    "seductive": {"speed": 0.80, "pitch": -15},    # Very slow, deep, maximum allure
    "intimate": {"speed": 0.82, "pitch": -12},     # Slow, breathy, close
    "flirty": {"speed": 0.85, "pitch": -10},       # Playful but alluring
    # Personality-based settings - all get seductive treatment
    "companion": {"speed": 0.82, "pitch": -12},    # Companion = seductive by default
    "therapist": {"speed": 0.85, "pitch": -8},     # Warm, intimate therapy voice
    "empathetic": {"speed": 0.85, "pitch": -8},    # Warm, caring, slightly alluring
    "friendly": {"speed": 0.88, "pitch": -5},      # Warm and approachable
    "calm": {"speed": 0.85, "pitch": -6},          # Slow, peaceful
    # Default
    "default": {"speed": 0.85, "pitch": -10}       # Default = intimate settings
}

# Voice configuration by accent - premium female voices
ACCENT_VOICES = {
    "en-us": "en-US-AvaNeural",        # Smooth, warm, natural
    "en-gb": "en-GB-SoniaNeural",      # Sophisticated British allure
    "en-au": "en-AU-NatashaNeural",    # Relaxed Australian warmth
    "en-newest": "en-US-AvaNeural",    # Best quality alluring US voice
    "en-calm": "en-US-AriaNeural",     # Calm, soothing meditation voice
    "en-warm": "en-US-AvaNeural",      # Warm, inviting tone
    "en-gentle": "en-US-SaraNeural",   # Soft, gentle voice
    "en-friendly": "en-US-JennyNeural" # Friendly, empathetic
}


class TTSRequest(BaseModel):
    """Request model for TTS synthesis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field(default="en-US-JennyNeural", description="Voice name")
    rate: str = Field(default="+0%", description="Speech rate (-50% to +100%)")
    pitch: str = Field(default="+0Hz", description="Pitch adjustment")


class VoiceInfo(BaseModel):
    """Voice information model"""
    name: str
    short_name: str
    gender: str
    locale: str
    language: str
    suggested_codec: Optional[str] = None


class TTSHealthResponse(BaseModel):
    """TTS health check response"""
    status: str
    version: str
    edge_tts_available: bool
    cache_enabled: bool
    voices_available: int
    timestamp: str
    service: str = "memOS TTS Server"


@router.get("/health")
async def health_check() -> TTSHealthResponse:
    """Health check endpoint for TTS service"""
    voice_count = 0
    if EDGE_TTS_AVAILABLE:
        try:
            voices = await edge_tts.list_voices()
            voice_count = len(voices)
        except Exception:
            pass

    return TTSHealthResponse(
        status="healthy" if EDGE_TTS_AVAILABLE else "degraded",
        version="2.0",
        edge_tts_available=EDGE_TTS_AVAILABLE,
        cache_enabled=True,
        voices_available=voice_count,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get("/speakers")
async def get_speakers() -> dict:
    """Get available speakers - MeloTTS compatible endpoint"""
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="edge-tts not available")

    try:
        voices = await edge_tts.list_voices()
        # Return English voices as speakers
        speakers = [v["ShortName"] for v in voices if v["Locale"].startswith("en-")]
        return {"speakers": speakers[:20]}  # Limit to 20 for performance
    except Exception as e:
        logger.error(f"Error getting speakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class MeloTTSRequest(BaseModel):
    """MeloTTS synthesis request model"""
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = Field(default="en-US-JennyNeural", alias="voice_id")
    sr: int = Field(default=22050, alias="sr")  # Sample rate
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    noise: float = Field(default=0.6)
    noisew: float = Field(default=0.8)
    length_scale: float = Field(default=1.0)
    format: str = Field(default="wav")


@router.post("/tts/generate")
async def generate_tts_post(request: MeloTTSRequest) -> Response:
    """Generate TTS audio - MeloTTS compatible endpoint"""
    # Use voice_id directly as the voice name (e.g., "en-US-AvaNeural")
    # Default to Ava - warm, naturally alluring female voice
    voice = request.voice_id if request.voice_id and "Neural" in request.voice_id else "en-US-AvaNeural"
    logger.info(f"TTS generate request: voice={voice}, speed={request.speed}, text_len={len(request.text)}")
    # Must pass pitch explicitly as int, not FastAPI Query default
    return await base_tts(text=request.text, accent=voice, speed=request.speed, pitch=0, language="English")


@router.get("/voices")
async def list_voices(
    locale: Optional[str] = Query(None, description="Filter by locale (e.g., en-US)")
) -> List[VoiceInfo]:
    """List all available TTS voices"""
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="edge-tts not available. Install with: pip install edge-tts"
        )

    try:
        voices = await edge_tts.list_voices()

        result = []
        for voice in voices:
            if locale and not voice["Locale"].lower().startswith(locale.lower()):
                continue

            result.append(VoiceInfo(
                name=voice.get("FriendlyName", voice["ShortName"]),
                short_name=voice["ShortName"],
                gender=voice.get("Gender", "Unknown"),
                locale=voice["Locale"],
                language=voice.get("Language", voice["Locale"]),
                suggested_codec=voice.get("SuggestedCodec")
            ))

        return result

    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/base_tts/")
async def base_tts(
    text: str = Query(..., description="Text to synthesize"),
    accent: str = Query("en-newest", description="Voice accent"),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="Speech speed"),
    pitch: int = Query(0, ge=-20, le=20, description="Pitch adjustment in Hz (-20 to +20)"),
    language: str = Query("English", description="Language")
) -> Response:
    """
    Base TTS endpoint - compatible with OpenVoice API format.
    Returns WAV audio data (16-bit PCM, mono) for Android AudioTrack compatibility.

    For seductive/intimate voice, use:
    - speed: 0.80-0.85 (slower, deliberate pacing)
    - pitch: -10 to -15 Hz (adds depth and allure)
    """
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="edge-tts not available. Install with: pip install edge-tts"
        )

    # Get voice for accent - support both accent keys and direct voice names
    if "Neural" in accent:
        # Direct voice name provided (e.g., "en-US-AvaNeural")
        voice = accent
    else:
        # Accent key provided (e.g., "en-us", "en-newest")
        voice = ACCENT_VOICES.get(accent.lower(), "en-US-AvaNeural")

    # Convert speed to rate percentage
    rate_percent = int((speed - 1.0) * 100)
    rate = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"

    # Convert pitch to Hz string
    pitch_str = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

    # Generate cache key - use .pcm suffix for raw PCM data (include pitch in key)
    cache_key = hashlib.md5(f"{text}:{voice}:{rate}:{pitch_str}:pcm".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.pcm"

    # Check cache for PCM file
    if cache_file.exists():
        file_age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if file_age_hours < CACHE_TTL_HOURS:
            logger.debug(f"Cache hit for TTS PCM: {cache_key}")
            audio_data = cache_file.read_bytes()
            return Response(
                content=audio_data,
                media_type="audio/L16;rate=22050;channels=1",  # Raw PCM format
                headers={
                    "Content-Length": str(len(audio_data)),
                    "Cache-Control": f"public, max-age={int(CACHE_TTL_HOURS * 3600)}",
                    "X-TTS-Cache": "hit",
                    "X-Audio-Format": "pcm",
                    "X-Audio-Sample-Rate": "22050",
                    "X-Audio-Channels": "1",
                    "X-Audio-Bits": "16"
                }
            )

    try:
        logger.info(f"Generating TTS: voice={voice}, rate={rate}, pitch={pitch_str}, text_len={len(text)}")

        # Generate speech (edge-tts produces MP3)
        # Include pitch for seductive/intimate voice effects
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch_str)

        # Collect audio bytes
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        mp3_data = b"".join(audio_chunks)
        logger.debug(f"Generated {len(mp3_data)} bytes of MP3 audio")

        # Convert MP3 to raw PCM for Android AudioTrack compatibility
        pcm_data = convert_mp3_to_pcm(mp3_data, sample_rate=22050)
        logger.info(f"Converted MP3 to PCM: {len(mp3_data)} -> {len(pcm_data)} bytes")

        # Cache the PCM result
        try:
            cache_file.write_bytes(pcm_data)
        except Exception as e:
            logger.warning(f"Failed to cache TTS PCM: {e}")

        return Response(
            content=pcm_data,
            media_type="audio/L16;rate=22050;channels=1",  # Raw PCM format
            headers={
                "Content-Length": str(len(pcm_data)),
                "Cache-Control": f"public, max-age={int(CACHE_TTL_HOURS * 3600)}",
                "X-TTS-Cache": "miss",
                "X-Audio-Format": "pcm",
                "X-Audio-Sample-Rate": "22050",
                "X-Audio-Channels": "1",
                "X-Audio-Bits": "16"
            }
        )

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@router.get("/synthesize_speech/")
async def synthesize_speech(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query("default", description="Voice personality"),
    accent: str = Query(None, description="Optional accent override or direct voice ID"),
    speed: float = Query(None, ge=0.5, le=2.0, description="Speech speed (None = auto based on personality)"),
    pitch: int = Query(None, ge=-20, le=20, description="Pitch in Hz (None = auto based on personality)"),
    emotion: str = Query(None, description="Emotion style (seductive, intimate, flirty, friendly, etc.)"),
    watermark: str = Query(None, description="Audio watermark (ignored)"),
    voice_id: str = Query(None, description="Direct voice ID (e.g., en-US-AvaNeural)")
) -> Response:
    """
    Synthesize speech with personality selection and seductive voice support.
    Compatible with OpenVoice API format.

    For seductive/intimate voice, use:
    - voice: "seductive", "intimate", or "flirty"
    - OR emotion: "seductive", "intimate", "flirty"

    Voice selection priority:
    1. voice_id parameter (direct voice like "en-US-AvaNeural")
    2. accent parameter if it's a Neural voice
    3. voice parameter mapped to DEFAULT_VOICES
    4. accent parameter mapped to ACCENT_VOICES
    5. Default to Natasha (warm, alluring Australian)

    Speed/Pitch auto-adjustment:
    - Seductive personalities automatically get slower speed and lower pitch
    - Can be overridden with explicit speed/pitch parameters
    """
    selected_voice = None

    # Priority 1: Direct voice_id parameter
    if voice_id and "Neural" in voice_id:
        selected_voice = voice_id
        logger.debug(f"Using direct voice_id: {voice_id}")

    # Priority 2: accent parameter if it's a Neural voice
    elif accent and "Neural" in accent:
        selected_voice = accent
        logger.debug(f"Using accent as voice: {accent}")

    # Priority 3: voice parameter mapped to DEFAULT_VOICES (personality names)
    elif voice in DEFAULT_VOICES:
        selected_voice = DEFAULT_VOICES[voice]
        logger.debug(f"Mapped personality '{voice}' to voice: {selected_voice}")

    # Priority 4: accent parameter mapped to ACCENT_VOICES
    elif accent and accent.lower() in ACCENT_VOICES:
        selected_voice = ACCENT_VOICES[accent.lower()]
        logger.debug(f"Mapped accent '{accent}' to voice: {selected_voice}")

    # Priority 5: Default to Natasha (Australian)
    else:
        selected_voice = "en-AU-NatashaNeural"
        logger.debug(f"Using default alluring voice: {selected_voice}")

    # Determine seductive settings based on personality or emotion
    # Priority: emotion > voice personality > default (all with seductive treatment)
    emotion_key = emotion.lower() if emotion else None
    personality_key = voice.lower() if voice else None

    # Check emotion first, then personality, then default
    seductive_config = (
        SEDUCTIVE_SETTINGS.get(emotion_key) if emotion_key else None
    ) or (
        SEDUCTIVE_SETTINGS.get(personality_key) if personality_key else None
    ) or SEDUCTIVE_SETTINGS.get("default", {"speed": 0.85, "pitch": -10})

    # Use explicit values if provided, otherwise use seductive config
    effective_speed = speed if speed is not None else seductive_config.get("speed", 1.0)
    effective_pitch = pitch if pitch is not None else seductive_config.get("pitch", 0)

    logger.info(f"synthesize_speech: voice={selected_voice}, personality={voice}, emotion={emotion}, speed={effective_speed}, pitch={effective_pitch}")

    # Delegate to base_tts with seductive settings
    return await base_tts(text=text, accent=selected_voice, speed=effective_speed, pitch=effective_pitch, language="English")


@router.post("/synthesize")
async def synthesize_post(request: TTSRequest) -> Response:
    """
    POST endpoint for TTS synthesis with full control.
    """
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="edge-tts not available"
        )

    try:
        communicate = edge_tts.Communicate(
            request.text,
            request.voice,
            rate=request.rate,
            pitch=request.pitch
        )

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        audio_data = b"".join(audio_chunks)

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Length": str(len(audio_data))}
        )

    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
async def stream_tts(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query("en-US-JennyNeural", description="Voice name"),
    rate: str = Query("+0%", description="Speech rate")
):
    """
    Stream TTS audio in real-time.
    Useful for long texts where you want to start playing before generation completes.
    """
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="edge-tts not available")

    async def audio_generator():
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")

    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg",
        headers={"Transfer-Encoding": "chunked"}
    )


@router.delete("/cache")
async def clear_cache() -> dict:
    """Clear the TTS cache"""
    try:
        count = 0
        for cache_file in CACHE_DIR.glob("*.mp3"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached TTS files")
        return {"success": True, "files_removed": count}

    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cleanup old cache files on module load
def _cleanup_old_cache():
    """Remove cache files older than TTL"""
    try:
        now = datetime.now().timestamp()
        for cache_file in CACHE_DIR.glob("*.mp3"):
            file_age_hours = (now - cache_file.stat().st_mtime) / 3600
            if file_age_hours > CACHE_TTL_HOURS:
                cache_file.unlink()
    except Exception:
        pass

_cleanup_old_cache()


# =============================================================================
# EmotiVoice and OpenVoice Integration (Apache 2.0 / MIT Licensed)
# =============================================================================

# Lazy import services to avoid loading models at startup
EMOTIVOICE_AVAILABLE = False
OPENVOICE_AVAILABLE = False

try:
    from services.emotivoice_service import get_emotivoice_service, EMOTIVOICE_EMOTIONS
    EMOTIVOICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"EmotiVoice not available: {e}")
    EMOTIVOICE_EMOTIONS = []

try:
    from services.openvoice_service import get_openvoice_service, OPENVOICE_STYLES
    OPENVOICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenVoice not available: {e}")
    OPENVOICE_STYLES = []


class EmotiVoiceRequest(BaseModel):
    """EmotiVoice TTS request model"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    emotion: str = Field(default="Neutral", description="Emotion prompt (Happy, Sad, Empathetic, etc.)")
    speaker: str = Field(default="8051", description="Speaker ID")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    format: str = Field(default="pcm", description="Output format: pcm, wav, mp3")


class OpenVoiceRequest(BaseModel):
    """OpenVoice TTS request model"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: Optional[str] = Field(default=None, description="Registered voice ID for cloning")
    style: str = Field(default="default", description="Voice style (friendly, cheerful, whispering, etc.)")
    language: str = Field(default="English", description="Language: English or Chinese")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    format: str = Field(default="pcm", description="Output format: pcm, wav, mp3")


class VoiceRegistrationRequest(BaseModel):
    """Voice registration request"""
    user_id: str = Field(..., description="User ID")
    voice_name: str = Field(default="default", description="Name for this voice")
    language: str = Field(default="en", description="Language code")


# -----------------------------------------------------------------------------
# EmotiVoice Endpoints
# -----------------------------------------------------------------------------

@router.get("/emotivoice/emotions")
async def get_emotivoice_emotions() -> dict:
    """Get list of available emotion prompts for EmotiVoice"""
    return {
        "available": EMOTIVOICE_AVAILABLE,
        "emotions": EMOTIVOICE_EMOTIONS if EMOTIVOICE_AVAILABLE else [],
        "description": "Use any emotion prompt text - these are recommendations"
    }


@router.get("/emotivoice/speakers")
async def get_emotivoice_speakers() -> dict:
    """Get available EmotiVoice speakers"""
    if not EMOTIVOICE_AVAILABLE:
        return {"available": False, "speakers": [], "presets": {}}

    service = get_emotivoice_service()
    if not service.is_loaded:
        await service.initialize()

    return {
        "available": True,
        "speakers": service.list_speakers()[:50],  # Limit to first 50
        "presets": service.get_speaker_presets()
    }


@router.post("/emotivoice/synthesize")
async def emotivoice_synthesize(request: EmotiVoiceRequest) -> Response:
    """
    Synthesize speech with EmotiVoice emotion control.

    Emotions are prompt-based - you can use any descriptive text:
    - "Happy", "Excited", "Joyful"
    - "Sad", "Melancholic", "Sorrowful"
    - "Empathetic", "Supportive", "Caring"
    - "Angry", "Frustrated"
    - Or any custom emotion description
    """
    if not EMOTIVOICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="EmotiVoice not available")

    try:
        service = get_emotivoice_service()
        audio_data = await service.synthesize(
            text=request.text,
            emotion=request.emotion,
            speaker=request.speaker,
            speed=request.speed,
            output_format=request.format
        )

        media_type = {
            "pcm": "audio/L16;rate=22050;channels=1",
            "wav": "audio/wav",
            "mp3": "audio/mpeg"
        }.get(request.format, "audio/L16;rate=22050;channels=1")

        return Response(
            content=audio_data,
            media_type=media_type,
            headers={
                "Content-Length": str(len(audio_data)),
                "X-TTS-Engine": "emotivoice",
                "X-TTS-Emotion": request.emotion,
                "X-Audio-Format": request.format,
                "X-Audio-Sample-Rate": "22050",
                "X-Audio-Channels": "1",
                "X-Audio-Bits": "16"
            }
        )

    except Exception as e:
        logger.error(f"EmotiVoice synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# OpenVoice Endpoints
# -----------------------------------------------------------------------------

@router.get("/openvoice/styles")
async def get_openvoice_styles() -> dict:
    """Get available voice styles for OpenVoice"""
    return {
        "available": OPENVOICE_AVAILABLE,
        "styles": OPENVOICE_STYLES if OPENVOICE_AVAILABLE else [],
        "description": "Styles control emotion/manner of speech"
    }


@router.get("/openvoice/voices")
async def get_openvoice_voices(user_id: Optional[str] = None) -> dict:
    """Get registered voices for OpenVoice cloning"""
    if not OPENVOICE_AVAILABLE:
        return {"available": False, "voices": []}

    service = get_openvoice_service()
    return {
        "available": True,
        "voices": service.list_voices(user_id)
    }


@router.post("/openvoice/register-voice")
async def register_openvoice_voice(
    user_id: str = Query(..., description="User ID"),
    voice_name: str = Query(default="default", description="Name for this voice"),
    language: str = Query(default="en", description="Language code"),
    audio_file: bytes = None
) -> dict:
    """
    Register a voice for cloning from audio sample.
    Audio should be 10-30 seconds of clear speech.

    Note: This endpoint expects audio data in the request body.
    Use multipart/form-data with an 'audio' field.
    """
    if not OPENVOICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenVoice not available")

    # For now, return instructions - full implementation needs file upload
    return {
        "success": False,
        "message": "Use POST /openvoice/register-voice-upload with multipart form data",
        "instructions": {
            "method": "POST",
            "content_type": "multipart/form-data",
            "fields": {
                "user_id": "Your user ID",
                "voice_name": "Name for the voice",
                "language": "en or zh",
                "audio": "Audio file (WAV, MP3) - 10-30 seconds"
            }
        }
    }


@router.post("/openvoice/synthesize")
async def openvoice_synthesize(request: OpenVoiceRequest) -> Response:
    """
    Synthesize speech with OpenVoice.

    If voice_id is provided, applies voice cloning.
    Style controls the emotion/manner of speech.

    Available styles:
    - default, friendly, cheerful, excited
    - sad, angry, terrified
    - shouting, whispering
    """
    if not OPENVOICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenVoice not available")

    try:
        service = get_openvoice_service()
        audio_data = await service.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            style=request.style,
            language=request.language,
            speed=request.speed,
            output_format=request.format
        )

        media_type = {
            "pcm": "audio/L16;rate=22050;channels=1",
            "wav": "audio/wav",
            "mp3": "audio/mpeg"
        }.get(request.format, "audio/L16;rate=22050;channels=1")

        return Response(
            content=audio_data,
            media_type=media_type,
            headers={
                "Content-Length": str(len(audio_data)),
                "X-TTS-Engine": "openvoice",
                "X-TTS-Style": request.style,
                "X-TTS-VoiceCloned": str(request.voice_id is not None),
                "X-Audio-Format": request.format,
                "X-Audio-Sample-Rate": "22050",
                "X-Audio-Channels": "1",
                "X-Audio-Bits": "16"
            }
        )

    except Exception as e:
        logger.error(f"OpenVoice synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Unified TTS Endpoint
# -----------------------------------------------------------------------------

class UnifiedTTSRequest(BaseModel):
    """Unified TTS request - auto-routes to best engine"""
    text: str = Field(..., min_length=1, max_length=5000)
    engine: Optional[str] = Field(default="auto", description="Engine: auto, edge-tts, emotivoice, openvoice")
    voice_id: Optional[str] = Field(default=None, description="Voice ID for cloning (OpenVoice)")
    emotion: Optional[str] = Field(default=None, description="Emotion prompt (EmotiVoice)")
    style: Optional[str] = Field(default=None, description="Style (OpenVoice)")
    personality: Optional[str] = Field(default=None, description="Personality preset (Edge-TTS)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = Field(default="pcm")


@router.post("/unified/synthesize")
async def unified_synthesize(request: UnifiedTTSRequest) -> Response:
    """
    Unified TTS endpoint - automatically routes to the best engine.

    Routing logic:
    - If voice_id provided → OpenVoice (voice cloning)
    - If emotion provided → EmotiVoice (emotion control)
    - If style provided → OpenVoice (style control)
    - Otherwise → Edge-TTS (fastest, personality support)
    """
    engine = request.engine

    # Auto-routing
    if engine == "auto":
        if request.voice_id:
            engine = "openvoice"
        elif request.emotion and EMOTIVOICE_AVAILABLE:
            engine = "emotivoice"
        elif request.style and OPENVOICE_AVAILABLE:
            engine = "openvoice"
        else:
            engine = "edge-tts"

    logger.info(f"Unified TTS: engine={engine}, text_len={len(request.text)}")

    # Route to appropriate engine
    if engine == "emotivoice":
        return await emotivoice_synthesize(EmotiVoiceRequest(
            text=request.text,
            emotion=request.emotion or "Neutral",
            speed=request.speed,
            format=request.format
        ))

    elif engine == "openvoice":
        return await openvoice_synthesize(OpenVoiceRequest(
            text=request.text,
            voice_id=request.voice_id,
            style=request.style or "default",
            speed=request.speed,
            format=request.format
        ))

    else:  # edge-tts
        return await synthesize_speech(
            text=request.text,
            voice=request.personality or "companion",
            speed=request.speed
        )


@router.get("/engines")
async def list_tts_engines() -> dict:
    """List all available TTS engines and their capabilities"""
    engines = {
        "edge-tts": {
            "available": EDGE_TTS_AVAILABLE,
            "license": "Microsoft Edge (free for personal use)",
            "features": ["322 neural voices", "personality presets", "speed/pitch control"],
            "best_for": "Fast synthesis, multiple accents"
        },
        "emotivoice": {
            "available": EMOTIVOICE_AVAILABLE,
            "license": "Apache 2.0 (commercial OK)",
            "features": ["2000+ voices", "prompt-based emotion", "EN/CN"],
            "best_for": "Emotion-controlled speech"
        },
        "openvoice": {
            "available": OPENVOICE_AVAILABLE,
            "license": "MIT (commercial OK)",
            "features": ["voice cloning", "style control", "multi-language"],
            "best_for": "Voice cloning, style variety"
        }
    }

    return {
        "engines": engines,
        "recommended": "edge-tts" if EDGE_TTS_AVAILABLE else (
            "emotivoice" if EMOTIVOICE_AVAILABLE else "openvoice"
        )
    }


# -----------------------------------------------------------------------------
# VRAM Management Endpoints
# -----------------------------------------------------------------------------

@router.post("/models/unload")
async def unload_tts_models(engine: str = Query("all", description="Engine to unload: all, emotivoice, openvoice")) -> dict:
    """
    Unload TTS models to free VRAM.
    Use this before running LLM inference for hot-swap VRAM management.
    """
    unloaded = []

    if engine in ["all", "emotivoice"] and EMOTIVOICE_AVAILABLE:
        service = get_emotivoice_service()
        if service.is_loaded:
            service.unload_models()
            unloaded.append("emotivoice")

    if engine in ["all", "openvoice"] and OPENVOICE_AVAILABLE:
        service = get_openvoice_service()
        if service.is_loaded:
            service.unload_models()
            unloaded.append("openvoice")

    return {
        "success": True,
        "unloaded": unloaded,
        "message": f"Unloaded {len(unloaded)} TTS engine(s) to free VRAM"
    }


@router.get("/models/status")
async def tts_models_status() -> dict:
    """Check which TTS models are currently loaded in VRAM"""
    status = {}

    if EMOTIVOICE_AVAILABLE:
        service = get_emotivoice_service()
        status["emotivoice"] = {
            "available": True,
            "loaded": service.is_loaded
        }
    else:
        status["emotivoice"] = {"available": False, "loaded": False}

    if OPENVOICE_AVAILABLE:
        service = get_openvoice_service()
        status["openvoice"] = {
            "available": True,
            "loaded": service.is_loaded
        }
    else:
        status["openvoice"] = {"available": False, "loaded": False}

    status["edge_tts"] = {
        "available": EDGE_TTS_AVAILABLE,
        "loaded": EDGE_TTS_AVAILABLE  # Edge-TTS doesn't use GPU
    }

    return status
