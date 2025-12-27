"""
EmotiVoice TTS Service - Emotion-Controlled Text-to-Speech
Apache 2.0 License - Free for commercial use

Supports:
- Prompt-based emotion control (happy, excited, sad, angry, etc.)
- 2000+ built-in voices
- English and Chinese
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
import io
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("memOS.emotivoice")

# Add EmotiVoice to path
EMOTIVOICE_DIR = Path(__file__).parent.parent.parent / "EmotiVoice"
sys.path.insert(0, str(EMOTIVOICE_DIR))

# Supported emotion prompts
EMOTIVOICE_EMOTIONS = [
    # English emotions
    "Happy", "Excited", "Surprised",
    "Sad", "Angry", "Disgusted", "Fearful",
    "Calm", "Serious", "Neutral",
    # Therapeutic emotions for Recovery Bot
    "Empathetic", "Supportive", "Encouraging",
    "Gentle", "Soothing", "Reassuring",
    "Warm", "Caring", "Understanding",
    # Custom prompts also supported - any text describing emotion
]

# Sample speaker IDs (EmotiVoice has 2000+ speakers)
# VERIFIED HIGH-QUALITY FEMALE VOICES from EmotiVoice wiki:
# - 8051: Maria Kasper - "Clear, soothing, expressive" (277 LibriVox audiobooks!)
# - 11614: Sylviamb - "Crisp, melodic, captivating"
# - 92: Cori Samuel - "Lively, expressive, energetic"
# - 3559: Kerry Hiles - "Soothing, clear, inviting"
# Additional recommended: 65, 102, 225, 1088, 1093
SAMPLE_SPEAKERS = {
    # PRIMARY CHARMING FEMALE VOICES (verified from wiki)
    "female_soothing": "8051",    # Maria Kasper - Clear, soothing, expressive (BEST for warm)
    "female_inviting": "3559",    # Kerry Hiles - Soothing, clear, inviting (BEST for seductive)
    "female_melodic": "11614",    # Sylviamb - Crisp, melodic, captivating
    "female_lively": "92",        # Cori Samuel - Lively, expressive, energetic
    # ADDITIONAL RECOMMENDED FEMALE VOICES
    "female_warm": "1088",        # Recommended warm female
    "female_gentle": "1093",      # Recommended gentle female
    "female_soft": "225",         # Recommended soft female
    "female_sweet": "102",        # Recommended sweet female
    "female_breathy": "65",       # Recommended breathy female
    # MALE VOICES
    "male_calm": "6097",
    "male_deep": "6671",
}


@dataclass
class EmotiVoiceConfig:
    """Configuration for EmotiVoice model"""
    sampling_rate: int = 16000
    output_directory: str = "outputs"
    bert_path: str = "WangZeJun/simbert-base-chinese"


class EmotiVoiceService:
    """
    EmotiVoice TTS Service with emotion control and hot-swap VRAM management.

    Usage:
        service = EmotiVoiceService()
        await service.initialize()

        # Synthesize with emotion
        audio = await service.synthesize(
            text="I'm here to help you through this.",
            emotion="Empathetic",
            speaker="8051",
            speed=1.0
        )

        # Cleanup to free VRAM
        service.unload_models()
    """

    def __init__(
        self,
        outputs_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.outputs_dir = Path(outputs_dir) if outputs_dir else EMOTIVOICE_DIR / "outputs"

        # Model references (lazy loaded)
        self.style_encoder = None
        self.generator = None
        self.tokenizer = None
        self.token2id = None
        self.speaker2id = None
        self.g2p = None
        self.lexicon = None

        self._initialized = False
        self.sample_rate = 16000

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded in VRAM"""
        return self.generator is not None

    async def initialize(self) -> bool:
        """Initialize and load models into VRAM"""
        if self._initialized:
            return True

        # Save original directory
        original_cwd = os.getcwd()

        try:
            logger.info("Loading EmotiVoice models...")

            # Change to EmotiVoice directory
            os.chdir(str(EMOTIVOICE_DIR))

            import importlib.util
            from transformers import AutoTokenizer
            from yacs import config as CONFIG
            import glob

            # Save and temporarily hide conflicting server modules
            saved_modules = {}
            for prefix in ['models', 'config']:
                if prefix in sys.modules:
                    saved_modules[prefix] = sys.modules.pop(prefix)
                # Also remove any submodules
                for key in list(sys.modules.keys()):
                    if key.startswith(f'{prefix}.'):
                        saved_modules[key] = sys.modules.pop(key)

            # Also save and remove server paths from sys.path (but keep site-packages)
            server_dir = str(Path(__file__).parent.parent)  # /memOS/server
            saved_paths = []
            new_sys_path = []
            emotivoice_str = str(EMOTIVOICE_DIR)
            for i, p in enumerate(sys.path):
                # Remove server-related paths and empty string (current dir)
                # but KEEP site-packages and standard library
                if 'site-packages' in p or '/usr/lib/' in p or '.zip' in p:
                    new_sys_path.append(p)
                elif p == server_dir or p == '' or 'memOS/server' in p:
                    saved_paths.append((i, p))
                else:
                    new_sys_path.append(p)

            # Replace sys.path with cleaned version, EmotiVoice at front
            sys.path.clear()
            sys.path.append(emotivoice_str)
            sys.path.extend(new_sys_path)

            try:
                # Now imports should work correctly
                from config.joint.config import Config
                from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
                from models.prompt_tts_modified.jets import JETSGenerator
                from models.prompt_tts_modified.simbert import StyleEncoder
            finally:
                # Restore server's paths and modules after imports
                sys.path.clear()
                sys.path.extend([p for _, p in sorted(saved_paths)] + new_sys_path)
                # Re-add EmotiVoice path since we'll still need it
                if emotivoice_str not in sys.path:
                    sys.path.insert(0, emotivoice_str)
                for key, mod in saved_modules.items():
                    sys.modules[key] = mod

            config = Config()
            self.config = config
            self.sample_rate = config.sampling_rate

            def scan_checkpoint(cp_dir, prefix, c=8):
                pattern = os.path.join(cp_dir, prefix + '?' * c)
                cp_list = glob.glob(pattern)
                if len(cp_list) == 0:
                    return None
                return sorted(cp_list)[-1]

            # Find checkpoints
            am_checkpoint_path = scan_checkpoint(
                f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')
            style_encoder_checkpoint_path = scan_checkpoint(
                f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)

            if not am_checkpoint_path or not style_encoder_checkpoint_path:
                logger.error("EmotiVoice checkpoints not found")
                return False

            # Load model config
            with open(config.model_config_path, 'r') as fin:
                conf = CONFIG.load_cfg(fin)

            conf.n_vocab = config.n_symbols
            conf.n_speaker = config.speaker_n_labels

            # Load style encoder
            self.style_encoder = StyleEncoder(config)
            model_CKPT = torch.load(style_encoder_checkpoint_path, map_location="cpu")
            model_ckpt = {}
            for key, value in model_CKPT['model'].items():
                new_key = key[7:]
                model_ckpt[new_key] = value
            self.style_encoder.load_state_dict(model_ckpt, strict=False)

            # Load generator
            self.generator = JETSGenerator(conf).to(self.device)
            model_CKPT = torch.load(am_checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(model_CKPT['generator'])
            self.generator.eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

            # Load token and speaker mappings
            with open(config.token_list_path, 'r') as f:
                self.token2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

            with open(config.speaker2id_path, encoding='utf-8') as f:
                self.speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

            # Load G2P for text processing
            self.lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
            self.g2p = G2p()
            self.g2p_cn_en = g2p_cn_en

            self._initialized = True
            logger.info(f"EmotiVoice models loaded. {len(self.speaker2id)} speakers available.")
            return True

        except Exception as e:
            logger.error(f"Failed to load EmotiVoice models: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Restore original directory
            os.chdir(original_cwd)

    def unload_models(self):
        """Unload models to free VRAM"""
        logger.info("Unloading EmotiVoice models...")

        self.style_encoder = None
        self.generator = None
        self.tokenizer = None
        self.token2id = None
        self.speaker2id = None
        self._initialized = False

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("EmotiVoice models unloaded")

    def _get_style_embedding(self, prompt: str) -> np.ndarray:
        """Get style embedding from emotion prompt"""
        prompt_tokens = self.tokenizer([prompt], return_tensors="pt")
        input_ids = prompt_tokens["input_ids"]
        token_type_ids = prompt_tokens["token_type_ids"]
        attention_mask = prompt_tokens["attention_mask"]

        with torch.no_grad():
            output = self.style_encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        style_embedding = output["pooled_output"].cpu().squeeze().numpy()
        return style_embedding

    async def synthesize(
        self,
        text: str,
        emotion: str = "Neutral",
        speaker: str = "8051",
        speed: float = 1.0,
        output_format: str = "pcm"
    ) -> bytes:
        """
        Synthesize speech with emotion control.

        Args:
            text: Text to synthesize
            emotion: Emotion prompt (e.g., "Happy", "Sad", "Empathetic")
            speaker: Speaker ID (default "8051")
            speed: Speech speed (0.5 - 2.0)
            output_format: "pcm", "wav", or "mp3"

        Returns:
            Audio bytes in requested format
        """
        if not self._initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize EmotiVoice")

        try:
            # Process text with G2P
            processed_text = self.g2p_cn_en(text, self.g2p, self.lexicon)

            # Get style embeddings
            style_embedding = self._get_style_embedding(emotion)
            content_embedding = self._get_style_embedding(text)

            # Get speaker ID
            if speaker not in self.speaker2id:
                speaker = list(self.speaker2id.keys())[0]
            speaker_id = self.speaker2id[speaker]

            # Convert text to token IDs
            text_int = [self.token2id[ph] for ph in processed_text.split() if ph in self.token2id]

            # Prepare tensors
            sequence = torch.from_numpy(np.array(text_int)).to(self.device).long().unsqueeze(0)
            sequence_len = torch.from_numpy(np.array([len(text_int)])).to(self.device)
            style_embedding = torch.from_numpy(style_embedding).to(self.device).unsqueeze(0)
            content_embedding = torch.from_numpy(content_embedding).to(self.device).unsqueeze(0)
            speaker_tensor = torch.from_numpy(np.array([speaker_id])).to(self.device)

            # Generate audio
            with torch.no_grad():
                infer_output = self.generator(
                    inputs_ling=sequence,
                    inputs_style_embedding=style_embedding,
                    input_lengths=sequence_len,
                    inputs_content_embedding=content_embedding,
                    inputs_speaker=speaker_tensor,
                    alpha=1.0
                )

            audio = infer_output["wav_predictions"].squeeze() * 32768.0
            audio = audio.cpu().numpy().astype('int16')

            # Apply speed adjustment if needed
            if speed != 1.0:
                import pyrubberband as pyrb
                audio_float = audio.astype(np.float32) / 32768.0
                audio_float = pyrb.time_stretch(audio_float, self.sample_rate, speed)
                audio = (audio_float * 32768.0).astype('int16')

            # Convert to requested format
            return self._convert_audio(audio, self.sample_rate, output_format)

        except Exception as e:
            logger.error(f"EmotiVoice synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _convert_audio(self, audio: np.ndarray, sample_rate: int, output_format: str) -> bytes:
        """Convert audio to requested format"""

        # Resample to 22050 for Android compatibility
        target_sr = 22050
        if sample_rate != target_sr:
            import librosa
            audio_float = audio.astype(np.float32) / 32768.0
            audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=target_sr)
            audio = (audio_float * 32768.0).astype(np.int16)

        if output_format == "pcm":
            # Raw 16-bit PCM for Android AudioTrack
            return audio.tobytes()

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

    def list_speakers(self) -> List[str]:
        """List available speaker IDs"""
        if self.speaker2id:
            return list(self.speaker2id.keys())
        return list(SAMPLE_SPEAKERS.values())

    def get_speaker_presets(self) -> Dict[str, str]:
        """Get named speaker presets"""
        return SAMPLE_SPEAKERS.copy()

    @staticmethod
    def get_available_emotions() -> List[str]:
        """Get list of recommended emotion prompts"""
        return EMOTIVOICE_EMOTIONS.copy()


# Singleton instance for hot-swap management
_emotivoice_service: Optional[EmotiVoiceService] = None

def get_emotivoice_service() -> EmotiVoiceService:
    """Get or create the EmotiVoice service singleton"""
    global _emotivoice_service
    if _emotivoice_service is None:
        _emotivoice_service = EmotiVoiceService()
    return _emotivoice_service
