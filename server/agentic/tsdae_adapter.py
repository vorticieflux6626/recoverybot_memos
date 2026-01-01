"""
TSDAE Domain Adapter for Unsupervised Embedding Adaptation

Implements TSDAE (Transformer-based Sequential Denoising Auto-Encoder) for domain
adaptation without labeled data. Based on Wang, Reimers, Gurevych (EMNLP 2021):
"TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for
Unsupervised Sentence Embedding Learning".

Key Features:
- Unsupervised domain adaptation using denoising autoencoder objective
- No labeled data needed - works with raw domain text
- Achieves up to 93.1% of supervised fine-tuning performance
- Requires only ~10K domain sentences for effective adaptation
- Supports incremental adaptation with new domain data

Research Reference:
- arXiv: https://arxiv.org/abs/2104.06979
- EMNLP 2021

Integration with memOS:
- Works with existing BGE-M3 and Qwen3-embedding models
- Outputs can be used with OptimalTransport fusion
- Supports FANUC robotics and industrial automation domains
"""

import asyncio
import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

# Lazy imports for optional dependencies
_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer, models
    from sentence_transformers import datasets as st_datasets
    from sentence_transformers import losses as st_losses
    from torch.utils.data import DataLoader
    import torch
    _sentence_transformers_available = True
except ImportError:
    pass

logger = logging.getLogger("agentic.tsdae_adapter")


class NoiseType(Enum):
    """Types of noise for denoising autoencoder."""
    DELETE = "delete"          # Random word deletion (default, 0.6 ratio)
    SWAP = "swap"              # Random word swapping
    INSERT = "insert"          # Random word insertion
    SUBSTITUTE = "substitute"  # Random word substitution


class PoolingMode(Enum):
    """Pooling strategies for sentence embeddings."""
    CLS = "cls"       # Use [CLS] token (default for TSDAE)
    MEAN = "mean"     # Mean pooling over tokens
    MAX = "max"       # Max pooling over tokens


class AdaptationStatus(Enum):
    """Status of domain adaptation."""
    NOT_STARTED = "not_started"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DomainConfig:
    """Configuration for a domain adaptation task."""
    domain_id: str
    domain_name: str
    base_model: str = "bert-base-uncased"  # Base model to adapt
    pooling_mode: PoolingMode = PoolingMode.CLS
    noise_type: NoiseType = NoiseType.DELETE
    noise_ratio: float = 0.6

    # Training parameters
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    warmup_steps: int = 100
    checkpoint_steps: int = 500

    # Tie encoder and decoder for parameter efficiency
    tie_encoder_decoder: bool = True

    # Output paths
    output_dir: str = "data/tsdae_models"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "domain_name": self.domain_name,
            "base_model": self.base_model,
            "pooling_mode": self.pooling_mode.value,
            "noise_type": self.noise_type.value,
            "noise_ratio": self.noise_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "checkpoint_steps": self.checkpoint_steps,
            "tie_encoder_decoder": self.tie_encoder_decoder,
            "output_dir": self.output_dir,
        }


@dataclass
class AdaptationResult:
    """Result of domain adaptation training."""
    domain_id: str
    status: AdaptationStatus
    model_path: Optional[str] = None
    training_time_seconds: float = 0.0
    num_sentences: int = 0
    epochs_completed: int = 0
    final_loss: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "status": self.status.value,
            "model_path": self.model_path,
            "training_time_seconds": self.training_time_seconds,
            "num_sentences": self.num_sentences,
            "epochs_completed": self.epochs_completed,
            "final_loss": self.final_loss,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class DomainEmbeddingResult:
    """Result of domain-adapted embedding."""
    embeddings: np.ndarray
    domain_id: str
    model_name: str
    dimension: int
    texts_embedded: int
    embedding_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "texts_embedded": self.texts_embedded,
            "embedding_time_ms": self.embedding_time_ms,
        }


class TSDaeAdapter:
    """
    TSDAE-based domain adapter for unsupervised embedding adaptation.

    This class manages domain-specific embedding models trained via the
    TSDAE objective (denoising autoencoder). It supports:
    - Training new domain adapters from raw text
    - Loading pre-trained domain adapters
    - Generating domain-adapted embeddings
    - Incremental updates with new domain data

    Example:
        adapter = TSDaeAdapter()

        # Add domain sentences
        fanuc_sentences = [
            "SRVO-063 BZAL alarm occurs during encoder initialization",
            "Replace pulsecoder if mastering calibration fails",
            ...
        ]

        # Train domain adapter
        config = DomainConfig(domain_id="fanuc", domain_name="FANUC Robotics")
        result = await adapter.train_adapter(fanuc_sentences, config)

        # Generate domain-adapted embeddings
        embeddings = await adapter.encode(["SRVO-063 alarm"], domain_id="fanuc")
    """

    def __init__(
        self,
        models_dir: str = "data/tsdae_models",
        device: Optional[str] = None,
    ):
        """
        Initialize TSDAE adapter.

        Args:
            models_dir: Directory for storing trained models
            device: PyTorch device ('cuda', 'cpu', or None for auto)
        """
        if not _sentence_transformers_available:
            raise ImportError(
                "sentence-transformers is required for TSDAE. "
                "Install with: pip install sentence-transformers"
            )

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Loaded domain models
        self._loaded_models: Dict[str, SentenceTransformer] = {}

        # Domain configurations
        self._domain_configs: Dict[str, DomainConfig] = {}

        # Training history
        self._training_history: List[AdaptationResult] = []

        # Statistics
        self._stats = {
            "domains_trained": 0,
            "total_sentences_processed": 0,
            "embeddings_generated": 0,
            "cache_hits": 0,
            "model_loads": 0,
        }

        logger.info(f"TSDaeAdapter initialized with device={self.device}")

    def _get_model_path(self, domain_id: str) -> Path:
        """Get the path for a domain model."""
        return self.models_dir / f"tsdae-{domain_id}"

    def _save_config(self, config: DomainConfig) -> None:
        """Save domain configuration to disk."""
        # Use domain_config.json to avoid conflicts with SentenceTransformer's config.json
        config_path = self._get_model_path(config.domain_id) / "domain_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def _load_config(self, domain_id: str) -> Optional[DomainConfig]:
        """Load domain configuration from disk."""
        config_path = self._get_model_path(domain_id) / "domain_config.json"
        if not config_path.exists():
            return None

        with open(config_path, "r") as f:
            data = json.load(f)

        return DomainConfig(
            domain_id=data["domain_id"],
            domain_name=data["domain_name"],
            base_model=data.get("base_model", "bert-base-uncased"),
            pooling_mode=PoolingMode(data.get("pooling_mode", "cls")),
            noise_type=NoiseType(data.get("noise_type", "delete")),
            noise_ratio=data.get("noise_ratio", 0.6),
            epochs=data.get("epochs", 1),
            batch_size=data.get("batch_size", 8),
            learning_rate=data.get("learning_rate", 3e-5),
            weight_decay=data.get("weight_decay", 0.0),
            warmup_steps=data.get("warmup_steps", 100),
            checkpoint_steps=data.get("checkpoint_steps", 500),
            tie_encoder_decoder=data.get("tie_encoder_decoder", True),
            output_dir=data.get("output_dir", "data/tsdae_models"),
        )

    async def train_adapter(
        self,
        sentences: List[str],
        config: DomainConfig,
        progress_callback: Optional[callable] = None,
    ) -> AdaptationResult:
        """
        Train a domain adapter using TSDAE objective.

        Args:
            sentences: List of domain-specific sentences (recommend 1K-100K)
            config: Domain configuration
            progress_callback: Optional callback for progress updates

        Returns:
            AdaptationResult with training outcome
        """
        if len(sentences) < 100:
            logger.warning(f"Few sentences ({len(sentences)}), recommend 1K+ for best results")

        start_time = time.time()
        model_path = self._get_model_path(config.domain_id)

        try:
            logger.info(f"Training TSDAE adapter for domain '{config.domain_id}' "
                       f"with {len(sentences)} sentences")

            # Create the sentence transformer model with specified pooling
            word_embedding_model = models.Transformer(config.base_model)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=config.pooling_mode.value,
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model],
                device=self.device,
            )

            # Create denoising dataset
            train_dataset = st_datasets.DenoisingAutoEncoderDataset(sentences)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )

            # Create denoising autoencoder loss
            train_loss = st_losses.DenoisingAutoEncoderLoss(
                model,
                decoder_name_or_path=config.base_model,
                tie_encoder_decoder=config.tie_encoder_decoder,
            )

            # Training callback for progress
            if progress_callback:
                class ProgressCallback:
                    def __init__(self, callback, total_steps):
                        self.callback = callback
                        self.total_steps = total_steps
                        self.current_step = 0

                    def __call__(self, score, epoch, steps):
                        self.current_step = steps
                        self.callback({
                            "epoch": epoch,
                            "steps": steps,
                            "total_steps": self.total_steps,
                            "progress": steps / self.total_steps if self.total_steps > 0 else 0,
                        })

                total_steps = len(train_dataloader) * config.epochs
                callback_fn = ProgressCallback(progress_callback, total_steps)
            else:
                callback_fn = None

            # Train the model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=config.epochs,
                weight_decay=config.weight_decay,
                scheduler="warmuplinear" if config.warmup_steps > 0 else "constantlr",
                warmup_steps=config.warmup_steps,
                optimizer_params={"lr": config.learning_rate},
                show_progress_bar=True,
                checkpoint_save_steps=config.checkpoint_steps,
                checkpoint_path=str(model_path / "checkpoints"),
                callback=callback_fn,
            )

            # Save the model
            model.save(str(model_path))
            self._save_config(config)

            # Update stats
            training_time = time.time() - start_time
            self._stats["domains_trained"] += 1
            self._stats["total_sentences_processed"] += len(sentences)

            # Store in loaded models
            self._loaded_models[config.domain_id] = model
            self._domain_configs[config.domain_id] = config

            result = AdaptationResult(
                domain_id=config.domain_id,
                status=AdaptationStatus.COMPLETED,
                model_path=str(model_path),
                training_time_seconds=training_time,
                num_sentences=len(sentences),
                epochs_completed=config.epochs,
            )

            self._training_history.append(result)
            logger.info(f"TSDAE training completed for '{config.domain_id}' in {training_time:.1f}s")

            return result

        except Exception as e:
            logger.error(f"TSDAE training failed for '{config.domain_id}': {e}")
            result = AdaptationResult(
                domain_id=config.domain_id,
                status=AdaptationStatus.FAILED,
                training_time_seconds=time.time() - start_time,
                num_sentences=len(sentences),
                error_message=str(e),
            )
            self._training_history.append(result)
            return result

    async def load_adapter(self, domain_id: str) -> bool:
        """
        Load a pre-trained domain adapter.

        Args:
            domain_id: Domain identifier

        Returns:
            True if loaded successfully, False otherwise
        """
        if domain_id in self._loaded_models:
            logger.debug(f"Domain '{domain_id}' already loaded")
            self._stats["cache_hits"] += 1
            return True

        model_path = self._get_model_path(domain_id)
        if not model_path.exists():
            logger.warning(f"No trained model found for domain '{domain_id}'")
            return False

        try:
            model = SentenceTransformer(str(model_path), device=self.device)
            config = self._load_config(domain_id)

            self._loaded_models[domain_id] = model
            if config:
                self._domain_configs[domain_id] = config

            self._stats["model_loads"] += 1
            logger.info(f"Loaded domain adapter for '{domain_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to load domain adapter '{domain_id}': {e}")
            return False

    async def encode(
        self,
        texts: Union[str, List[str]],
        domain_id: str,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> DomainEmbeddingResult:
        """
        Generate domain-adapted embeddings.

        Args:
            texts: Text or list of texts to embed
            domain_id: Domain to use for embedding
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar for large batches

        Returns:
            DomainEmbeddingResult with embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Ensure domain is loaded
        if domain_id not in self._loaded_models:
            loaded = await self.load_adapter(domain_id)
            if not loaded:
                raise ValueError(f"Domain '{domain_id}' not found. Train first with train_adapter()")

        model = self._loaded_models[domain_id]
        config = self._domain_configs.get(domain_id)

        start_time = time.time()

        # Generate embeddings
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embedding_time = (time.time() - start_time) * 1000  # Convert to ms

        self._stats["embeddings_generated"] += len(texts)

        return DomainEmbeddingResult(
            embeddings=embeddings,
            domain_id=domain_id,
            model_name=config.base_model if config else "unknown",
            dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
            texts_embedded=len(texts),
            embedding_time_ms=embedding_time,
        )

    def list_domains(self) -> List[Dict[str, Any]]:
        """List all available domain adapters."""
        domains = []

        # Check disk for saved models
        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("tsdae-"):
                    domain_id = model_dir.name.replace("tsdae-", "")
                    config = self._load_config(domain_id)

                    domains.append({
                        "domain_id": domain_id,
                        "domain_name": config.domain_name if config else domain_id,
                        "loaded": domain_id in self._loaded_models,
                        "model_path": str(model_dir),
                        "base_model": config.base_model if config else "unknown",
                    })

        return domains

    def unload_domain(self, domain_id: str) -> bool:
        """
        Unload a domain model to free memory.

        Args:
            domain_id: Domain to unload

        Returns:
            True if unloaded, False if not loaded
        """
        if domain_id in self._loaded_models:
            del self._loaded_models[domain_id]
            if domain_id in self._domain_configs:
                del self._domain_configs[domain_id]
            logger.info(f"Unloaded domain '{domain_id}'")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "loaded_domains": list(self._loaded_models.keys()),
            "device": self.device,
            "models_dir": str(self.models_dir),
            "training_history_count": len(self._training_history),
        }

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return [r.to_dict() for r in self._training_history]


class MultiDomainAdapter:
    """
    Multi-domain adapter that combines embeddings from multiple domains.

    Supports fusion strategies:
    - concatenation: [domain1_emb | domain2_emb]
    - mean: (domain1_emb + domain2_emb) / 2
    - weighted: w1 * domain1_emb + w2 * domain2_emb
    - attention: softmax(query · domain_embs) weighted sum

    Example:
        multi = MultiDomainAdapter()
        multi.add_domain("fanuc", fanuc_adapter)
        multi.add_domain("siemens", siemens_adapter)

        # Fused embedding covering both domains
        result = await multi.encode("servo alarm", domains=["fanuc", "siemens"])
    """

    def __init__(self):
        """Initialize multi-domain adapter."""
        self._adapters: Dict[str, TSDaeAdapter] = {}
        self._domain_weights: Dict[str, float] = {}

    def add_domain(
        self,
        domain_id: str,
        adapter: TSDaeAdapter,
        weight: float = 1.0,
    ) -> None:
        """Add a domain adapter."""
        self._adapters[domain_id] = adapter
        self._domain_weights[domain_id] = weight

    def remove_domain(self, domain_id: str) -> bool:
        """Remove a domain adapter."""
        if domain_id in self._adapters:
            del self._adapters[domain_id]
            del self._domain_weights[domain_id]
            return True
        return False

    async def encode(
        self,
        texts: Union[str, List[str]],
        domains: Optional[List[str]] = None,
        fusion: str = "mean",
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate fused embeddings from multiple domains.

        Args:
            texts: Text or list of texts
            domains: Domains to use (None = all)
            fusion: Fusion strategy ('concat', 'mean', 'weighted')
            normalize: Normalize final embeddings

        Returns:
            Dict with fused embeddings and per-domain results
        """
        if isinstance(texts, str):
            texts = [texts]

        if domains is None:
            domains = list(self._adapters.keys())

        if not domains:
            raise ValueError("No domains available")

        # Get embeddings from each domain
        domain_results = {}
        embeddings_list = []
        weights = []

        for domain_id in domains:
            if domain_id not in self._adapters:
                logger.warning(f"Domain '{domain_id}' not found, skipping")
                continue

            adapter = self._adapters[domain_id]
            result = await adapter.encode(texts, domain_id, normalize=False)

            domain_results[domain_id] = result.to_dict()
            embeddings_list.append(result.embeddings)
            weights.append(self._domain_weights.get(domain_id, 1.0))

        if not embeddings_list:
            raise ValueError("No valid domains found")

        # Fuse embeddings
        if fusion == "concat":
            fused = np.concatenate(embeddings_list, axis=1)
        elif fusion == "mean":
            fused = np.mean(embeddings_list, axis=0)
        elif fusion == "weighted":
            weights = np.array(weights) / sum(weights)
            fused = sum(w * e for w, e in zip(weights, embeddings_list))
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion}")

        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            fused = fused / norms

        return {
            "fused_embeddings": fused,
            "fusion_strategy": fusion,
            "domains_used": domains,
            "dimension": fused.shape[1] if len(fused.shape) > 1 else len(fused),
            "texts_embedded": len(texts),
            "per_domain_results": domain_results,
        }

    def list_domains(self) -> List[Dict[str, Any]]:
        """List all registered domains."""
        return [
            {
                "domain_id": domain_id,
                "weight": self._domain_weights[domain_id],
                "stats": adapter.get_stats(),
            }
            for domain_id, adapter in self._adapters.items()
        ]


# Predefined domain configurations
FANUC_DOMAIN_CONFIG = DomainConfig(
    domain_id="fanuc",
    domain_name="FANUC Robotics",
    base_model="bert-base-uncased",
    pooling_mode=PoolingMode.CLS,
    noise_type=NoiseType.DELETE,
    noise_ratio=0.6,
    epochs=1,
    batch_size=8,
    learning_rate=3e-5,
)

SIEMENS_DOMAIN_CONFIG = DomainConfig(
    domain_id="siemens",
    domain_name="Siemens PLC",
    base_model="bert-base-uncased",
    pooling_mode=PoolingMode.CLS,
    noise_type=NoiseType.DELETE,
    noise_ratio=0.6,
    epochs=1,
    batch_size=8,
    learning_rate=3e-5,
)

ROCKWELL_DOMAIN_CONFIG = DomainConfig(
    domain_id="rockwell",
    domain_name="Rockwell/Allen-Bradley",
    base_model="bert-base-uncased",
    pooling_mode=PoolingMode.CLS,
    noise_type=NoiseType.DELETE,
    noise_ratio=0.6,
    epochs=1,
    batch_size=8,
    learning_rate=3e-5,
)


# Factory function
def get_tsdae_adapter(models_dir: str = "data/tsdae_models") -> TSDaeAdapter:
    """Get a TSDAE adapter instance."""
    return TSDaeAdapter(models_dir=models_dir)


def get_multi_domain_adapter() -> MultiDomainAdapter:
    """Get a multi-domain adapter instance."""
    return MultiDomainAdapter()
