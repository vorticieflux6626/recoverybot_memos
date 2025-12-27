"""
OpenVoice TTS Service - Voice Cloning with Style Control
MIT License - Free for commercial use

Supports:
- Voice cloning from 10-30 second reference audio
- Style control: friendly, cheerful, excited, sad, angry, terrified, shouting, whispering
- Multi-language: English, Chinese
- Hot-swap VRAM management
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import io

logger = logging.getLogger("memOS.openvoice")

# Add OpenVoice to path
OPENVOICE_DIR = Path(__file__).parent.parent.parent / "OpenVoice"
sys.path.insert(0, str(OPENVOICE_DIR))

# Available styles in OpenVoice V1
OPENVOICE_STYLES = [
    "default",
    "friendly",
    "cheerful",
    "excited",
    "sad",
    "angry",
    "terrified",
    "shouting",
    "whispering"
]

@dataclass
class VoiceEmbedding:
    """Stored voice embedding for cloning"""
    user_id: str
    voice_name: str
    embedding: torch.Tensor
    created_at: float
    language: str = "en"


class OpenVoiceService:
    """
    OpenVoice TTS Service with hot-swap VRAM management.

    Usage:
        service = OpenVoiceService()
        await service.initialize()

        # Register a voice
        voice_id = await service.register_voice(user_id, audio_bytes, "my_voice")

        # Synthesize with cloned voice
        audio = await service.synthesize(
            text="Hello world",
            voice_id=voice_id,
            style="friendly",
            speed=1.0
        )

        # Cleanup to free VRAM
        service.unload_models()
    """

    def __init__(
        self,
        checkpoints_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else OPENVOICE_DIR / "checkpoints"

        # Model references (lazy loaded)
        self.base_tts_en = None
        self.base_tts_zh = None
        self.tone_converter = None

        # Source speaker embeddings (loaded from checkpoints)
        self.source_se_default = None
        self.source_se_style = None
        self.source_se_zh = None

        # Registered voice embeddings
        self.registered_voices: Dict[str, VoiceEmbedding] = {}

        self._initialized = False
        self.sample_rate = 22050

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded in VRAM"""
        return self.base_tts_en is not None

    async def initialize(self) -> bool:
        """Initialize and load models into VRAM"""
        if self._initialized:
            return True

        try:
            logger.info("Loading OpenVoice models...")

            # Import OpenVoice modules
            from openvoice.api import BaseSpeakerTTS, ToneColorConverter
            from openvoice import se_extractor

            self.se_extractor = se_extractor

            # Load English base TTS
            ckpt_en = self.checkpoints_dir / "base_speakers" / "EN"
            self.base_tts_en = BaseSpeakerTTS(
                str(ckpt_en / "config.json"),
                device=self.device
            )
            self.base_tts_en.load_ckpt(str(ckpt_en / "checkpoint.pth"))

            # Load Chinese base TTS
            ckpt_zh = self.checkpoints_dir / "base_speakers" / "ZH"
            if ckpt_zh.exists():
                self.base_tts_zh = BaseSpeakerTTS(
                    str(ckpt_zh / "config.json"),
                    device=self.device
                )
                self.base_tts_zh.load_ckpt(str(ckpt_zh / "checkpoint.pth"))

            # Load Tone Color Converter
            ckpt_converter = self.checkpoints_dir / "converter"
            self.tone_converter = ToneColorConverter(
                str(ckpt_converter / "config.json"),
                device=self.device,
                enable_watermark=False  # Disable watermark for cleaner audio
            )
            self.tone_converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))

            # Load source speaker embeddings
            self.source_se_default = torch.load(
                str(ckpt_en / "en_default_se.pth"),
                map_location=self.device
            )
            self.source_se_style = torch.load(
                str(ckpt_en / "en_style_se.pth"),
                map_location=self.device
            )
            if ckpt_zh.exists():
                self.source_se_zh = torch.load(
                    str(ckpt_zh / "zh_default_se.pth"),
                    map_location=self.device
                )

            self._initialized = True
            logger.info("OpenVoice models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load OpenVoice models: {e}")
            return False

    def unload_models(self):
        """Unload models to free VRAM"""
        logger.info("Unloading OpenVoice models...")

        self.base_tts_en = None
        self.base_tts_zh = None
        self.tone_converter = None
        self.source_se_default = None
        self.source_se_style = None
        self.source_se_zh = None
        self._initialized = False

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("OpenVoice models unloaded")

    async def register_voice(
        self,
        user_id: str,
        audio_data: bytes,
        voice_name: str = "default",
        language: str = "en"
    ) -> str:
        """
        Register a voice from audio sample for cloning.

        Args:
            user_id: User identifier
            audio_data: Audio bytes (WAV, MP3, etc.) - 10-30 seconds recommended
            voice_name: Name for this voice
            language: Language code (en, zh)

        Returns:
            voice_id: Unique identifier for this registered voice
        """
        if not self._initialized:
            await self.initialize()

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # Extract tone color embedding
            target_se, _ = self.se_extractor.get_se(
                temp_path,
                self.tone_converter,
                target_dir=str(OPENVOICE_DIR / "processed"),
                vad=True
            )

            # Generate voice ID
            import time
            voice_id = f"{user_id}_{voice_name}_{int(time.time())}"

            # Store embedding
            self.registered_voices[voice_id] = VoiceEmbedding(
                user_id=user_id,
                voice_name=voice_name,
                embedding=target_se,
                created_at=time.time(),
                language=language
            )

            logger.info(f"Registered voice: {voice_id}")
            return voice_id

        finally:
            os.unlink(temp_path)

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        style: str = "default",
        language: str = "English",
        speed: float = 1.0,
        output_format: str = "pcm"
    ) -> bytes:
        """
        Synthesize speech with optional voice cloning and style control.

        Args:
            text: Text to synthesize
            voice_id: Registered voice ID for cloning (None for default voice)
            style: One of OPENVOICE_STYLES
            language: "English" or "Chinese"
            speed: Speech speed (0.5 - 2.0)
            output_format: "pcm", "wav", or "mp3"

        Returns:
            Audio bytes in requested format
        """
        if not self._initialized:
            await self.initialize()

        # Select base TTS and source embedding
        if language.lower() in ["chinese", "zh"]:
            base_tts = self.base_tts_zh or self.base_tts_en
            source_se = self.source_se_zh or self.source_se_default
            lang_param = "Chinese"
        else:
            base_tts = self.base_tts_en
            source_se = self.source_se_style if style != "default" else self.source_se_default
            lang_param = "English"

        # Generate base audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            base_path = f.name

        try:
            # Run base TTS with style
            speaker = style if style in OPENVOICE_STYLES else "default"
            base_tts.tts(text, base_path, speaker=speaker, language=lang_param, speed=speed)

            # Apply voice cloning if voice_id provided
            if voice_id and voice_id in self.registered_voices:
                target_se = self.registered_voices[voice_id].embedding

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    output_path = f.name

                try:
                    self.tone_converter.convert(
                        audio_src_path=base_path,
                        src_se=source_se,
                        tgt_se=target_se,
                        output_path=output_path,
                        tau=0.3
                    )
                    audio_data, sr = sf.read(output_path)
                finally:
                    os.unlink(output_path)
            else:
                # Use base audio without cloning
                audio_data, sr = sf.read(base_path)

            # Convert to requested format
            return self._convert_audio(audio_data, sr, output_format)

        finally:
            os.unlink(base_path)

    def _convert_audio(self, audio: np.ndarray, sample_rate: int, output_format: str) -> bytes:
        """Convert audio to requested format"""

        # Resample to 22050 for Android compatibility
        target_sr = 22050
        if sample_rate != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)

        if output_format == "pcm":
            # Raw 16-bit PCM for Android AudioTrack
            audio_int16 = (audio * 32767).astype(np.int16)
            return audio_int16.tobytes()

        elif output_format == "wav":
            buffer = io.BytesIO()
            sf.write(buffer, audio, target_sr, format="WAV", subtype="PCM_16")
            return buffer.getvalue()

        elif output_format == "mp3":
            from pydub import AudioSegment
            buffer = io.BytesIO()
            sf.write(buffer, audio, target_sr, format="WAV")
            buffer.seek(0)
            wav_audio = AudioSegment.from_wav(buffer)
            mp3_buffer = io.BytesIO()
            wav_audio.export(mp3_buffer, format="mp3")
            return mp3_buffer.getvalue()

        return audio.tobytes()

    def list_voices(self, user_id: Optional[str] = None) -> List[dict]:
        """List registered voices, optionally filtered by user"""
        voices = []
        for voice_id, embedding in self.registered_voices.items():
            if user_id is None or embedding.user_id == user_id:
                voices.append({
                    "voice_id": voice_id,
                    "voice_name": embedding.voice_name,
                    "user_id": embedding.user_id,
                    "language": embedding.language,
                    "created_at": embedding.created_at
                })
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a registered voice"""
        if voice_id in self.registered_voices:
            del self.registered_voices[voice_id]
            logger.info(f"Deleted voice: {voice_id}")
            return True
        return False

    @staticmethod
    def get_available_styles() -> List[str]:
        """Get list of available voice styles"""
        return OPENVOICE_STYLES.copy()


# Singleton instance for hot-swap management
_openvoice_service: Optional[OpenVoiceService] = None

def get_openvoice_service() -> OpenVoiceService:
    """Get or create the OpenVoice service singleton"""
    global _openvoice_service
    if _openvoice_service is None:
        _openvoice_service = OpenVoiceService()
    return _openvoice_service
