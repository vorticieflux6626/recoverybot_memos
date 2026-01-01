"""
Unit tests for TSDAE Domain Adapter (G.7.4)

Tests the TSDAE-based domain adaptation module for unsupervised
embedding adaptation without labeled data.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from agentic.tsdae_adapter import (
    TSDaeAdapter,
    MultiDomainAdapter,
    DomainConfig,
    AdaptationResult,
    AdaptationStatus,
    DomainEmbeddingResult,
    NoiseType,
    PoolingMode,
    FANUC_DOMAIN_CONFIG,
    SIEMENS_DOMAIN_CONFIG,
    ROCKWELL_DOMAIN_CONFIG,
    get_tsdae_adapter,
    get_multi_domain_adapter,
)


class TestDomainConfig:
    """Tests for DomainConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DomainConfig(
            domain_id="test",
            domain_name="Test Domain",
        )
        assert config.domain_id == "test"
        assert config.domain_name == "Test Domain"
        assert config.base_model == "bert-base-uncased"
        assert config.pooling_mode == PoolingMode.CLS
        assert config.noise_type == NoiseType.DELETE
        assert config.noise_ratio == 0.6
        assert config.epochs == 1
        assert config.batch_size == 8
        assert config.learning_rate == 3e-5
        assert config.tie_encoder_decoder is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DomainConfig(
            domain_id="custom",
            domain_name="Custom Domain",
            base_model="distilbert-base-uncased",
            pooling_mode=PoolingMode.MEAN,
            noise_type=NoiseType.SWAP,
            noise_ratio=0.4,
            epochs=2,
            batch_size=16,
            learning_rate=5e-5,
        )
        assert config.base_model == "distilbert-base-uncased"
        assert config.pooling_mode == PoolingMode.MEAN
        assert config.noise_type == NoiseType.SWAP
        assert config.noise_ratio == 0.4
        assert config.epochs == 2
        assert config.batch_size == 16
        assert config.learning_rate == 5e-5

    def test_config_to_dict(self):
        """Test serialization to dictionary."""
        config = DomainConfig(
            domain_id="test",
            domain_name="Test Domain",
        )
        d = config.to_dict()
        assert d["domain_id"] == "test"
        assert d["domain_name"] == "Test Domain"
        assert d["pooling_mode"] == "cls"
        assert d["noise_type"] == "delete"
        assert "epochs" in d
        assert "batch_size" in d


class TestPredefinedConfigs:
    """Tests for predefined domain configurations."""

    def test_fanuc_config(self):
        """Test FANUC domain configuration."""
        assert FANUC_DOMAIN_CONFIG.domain_id == "fanuc"
        assert FANUC_DOMAIN_CONFIG.domain_name == "FANUC Robotics"
        assert FANUC_DOMAIN_CONFIG.base_model == "bert-base-uncased"
        assert FANUC_DOMAIN_CONFIG.pooling_mode == PoolingMode.CLS

    def test_siemens_config(self):
        """Test Siemens domain configuration."""
        assert SIEMENS_DOMAIN_CONFIG.domain_id == "siemens"
        assert SIEMENS_DOMAIN_CONFIG.domain_name == "Siemens PLC"

    def test_rockwell_config(self):
        """Test Rockwell/Allen-Bradley domain configuration."""
        assert ROCKWELL_DOMAIN_CONFIG.domain_id == "rockwell"
        assert ROCKWELL_DOMAIN_CONFIG.domain_name == "Rockwell/Allen-Bradley"


class TestAdaptationResult:
    """Tests for AdaptationResult dataclass."""

    def test_successful_result(self):
        """Test successful adaptation result."""
        result = AdaptationResult(
            domain_id="test",
            status=AdaptationStatus.COMPLETED,
            model_path="/path/to/model",
            training_time_seconds=120.5,
            num_sentences=1000,
            epochs_completed=1,
            final_loss=0.15,
        )
        assert result.status == AdaptationStatus.COMPLETED
        assert result.training_time_seconds == 120.5
        assert result.num_sentences == 1000
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed adaptation result."""
        result = AdaptationResult(
            domain_id="test",
            status=AdaptationStatus.FAILED,
            error_message="Out of memory",
        )
        assert result.status == AdaptationStatus.FAILED
        assert result.error_message == "Out of memory"

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        result = AdaptationResult(
            domain_id="test",
            status=AdaptationStatus.COMPLETED,
            num_sentences=500,
        )
        d = result.to_dict()
        assert d["domain_id"] == "test"
        assert d["status"] == "completed"
        assert d["num_sentences"] == 500
        assert "timestamp" in d


class TestEnums:
    """Tests for enumeration types."""

    def test_noise_types(self):
        """Test noise type values."""
        assert NoiseType.DELETE.value == "delete"
        assert NoiseType.SWAP.value == "swap"
        assert NoiseType.INSERT.value == "insert"
        assert NoiseType.SUBSTITUTE.value == "substitute"

    def test_pooling_modes(self):
        """Test pooling mode values."""
        assert PoolingMode.CLS.value == "cls"
        assert PoolingMode.MEAN.value == "mean"
        assert PoolingMode.MAX.value == "max"

    def test_adaptation_status(self):
        """Test adaptation status values."""
        assert AdaptationStatus.NOT_STARTED.value == "not_started"
        assert AdaptationStatus.TRAINING.value == "training"
        assert AdaptationStatus.COMPLETED.value == "completed"
        assert AdaptationStatus.FAILED.value == "failed"


class TestTSDaeAdapterInit:
    """Tests for TSDaeAdapter initialization."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for models."""
        temp_dir = tempfile.mkdtemp(prefix="tsdae_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_init_default(self, temp_models_dir):
        """Test default initialization."""
        adapter = TSDaeAdapter(models_dir=temp_models_dir)
        assert adapter.models_dir == Path(temp_models_dir)
        assert adapter.device in ["cuda", "cpu"]
        assert adapter._loaded_models == {}
        assert adapter._domain_configs == {}

    def test_init_with_device(self, temp_models_dir):
        """Test initialization with specific device."""
        adapter = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")
        assert adapter.device == "cpu"

    def test_factory_function(self):
        """Test factory function."""
        adapter = get_tsdae_adapter()
        assert isinstance(adapter, TSDaeAdapter)


class TestTSDaeAdapterMethods:
    """Tests for TSDaeAdapter methods (non-training)."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for models."""
        temp_dir = tempfile.mkdtemp(prefix="tsdae_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def adapter(self, temp_models_dir):
        """Create adapter instance."""
        return TSDaeAdapter(models_dir=temp_models_dir, device="cpu")

    def test_list_domains_empty(self, adapter):
        """Test listing domains when none exist."""
        domains = adapter.list_domains()
        assert domains == []

    def test_get_stats_initial(self, adapter):
        """Test initial statistics."""
        stats = adapter.get_stats()
        assert stats["domains_trained"] == 0
        assert stats["total_sentences_processed"] == 0
        assert stats["embeddings_generated"] == 0
        assert stats["loaded_domains"] == []

    def test_get_training_history_empty(self, adapter):
        """Test empty training history."""
        history = adapter.get_training_history()
        assert history == []

    def test_unload_nonexistent_domain(self, adapter):
        """Test unloading a domain that doesn't exist."""
        result = adapter.unload_domain("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_nonexistent_domain(self, adapter):
        """Test loading a domain that doesn't exist."""
        result = await adapter.load_adapter("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_encode_without_training(self, adapter):
        """Test encoding without training first."""
        with pytest.raises(ValueError, match="not found"):
            await adapter.encode("test text", domain_id="nonexistent")

    def test_model_path_generation(self, adapter):
        """Test model path generation."""
        path = adapter._get_model_path("test_domain")
        assert "tsdae-test_domain" in str(path)


class TestMultiDomainAdapter:
    """Tests for MultiDomainAdapter."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for models."""
        temp_dir = tempfile.mkdtemp(prefix="tsdae_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_init(self):
        """Test initialization."""
        multi = MultiDomainAdapter()
        assert multi._adapters == {}
        assert multi._domain_weights == {}

    def test_factory_function(self):
        """Test factory function."""
        multi = get_multi_domain_adapter()
        assert isinstance(multi, MultiDomainAdapter)

    def test_add_domain(self, temp_models_dir):
        """Test adding a domain."""
        multi = MultiDomainAdapter()
        adapter = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")
        multi.add_domain("test", adapter, weight=1.5)

        assert "test" in multi._adapters
        assert multi._domain_weights["test"] == 1.5

    def test_remove_domain(self, temp_models_dir):
        """Test removing a domain."""
        multi = MultiDomainAdapter()
        adapter = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")
        multi.add_domain("test", adapter)

        result = multi.remove_domain("test")
        assert result is True
        assert "test" not in multi._adapters

    def test_remove_nonexistent_domain(self):
        """Test removing a domain that doesn't exist."""
        multi = MultiDomainAdapter()
        result = multi.remove_domain("nonexistent")
        assert result is False

    def test_list_domains(self, temp_models_dir):
        """Test listing domains."""
        multi = MultiDomainAdapter()
        adapter = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")
        multi.add_domain("domain1", adapter, weight=1.0)
        multi.add_domain("domain2", adapter, weight=2.0)

        domains = multi.list_domains()
        assert len(domains) == 2
        domain_ids = [d["domain_id"] for d in domains]
        assert "domain1" in domain_ids
        assert "domain2" in domain_ids

    @pytest.mark.asyncio
    async def test_encode_no_domains(self):
        """Test encoding with no domains."""
        multi = MultiDomainAdapter()
        with pytest.raises(ValueError, match="No domains available"):
            await multi.encode("test text")


class TestDomainEmbeddingResult:
    """Tests for DomainEmbeddingResult dataclass."""

    def test_basic_result(self):
        """Test basic embedding result."""
        embeddings = np.random.randn(2, 768).astype(np.float32)
        result = DomainEmbeddingResult(
            embeddings=embeddings,
            domain_id="test",
            model_name="bert-base-uncased",
            dimension=768,
            texts_embedded=2,
            embedding_time_ms=15.5,
        )
        assert result.domain_id == "test"
        assert result.dimension == 768
        assert result.texts_embedded == 2
        assert result.embedding_time_ms == 15.5
        assert result.embeddings.shape == (2, 768)

    def test_result_to_dict(self):
        """Test serialization to dictionary."""
        embeddings = np.random.randn(3, 384).astype(np.float32)
        result = DomainEmbeddingResult(
            embeddings=embeddings,
            domain_id="test",
            model_name="distilbert",
            dimension=384,
            texts_embedded=3,
            embedding_time_ms=10.0,
        )
        d = result.to_dict()
        assert d["domain_id"] == "test"
        assert d["dimension"] == 384
        assert d["texts_embedded"] == 3
        # embeddings should not be in dict (too large)
        assert "embeddings" not in d


# Training tests (optional, requires GPU and time)
class TestTSDaeTraining:
    """
    Training tests for TSDAE adapter.

    These tests actually train models and require:
    - sentence-transformers installed
    - CUDA or sufficient CPU time
    - ~1-2 minutes per test

    Skip in CI with: pytest -m "not slow"
    """

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for models."""
        temp_dir = tempfile.mkdtemp(prefix="tsdae_train_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_sentences(self):
        """Sample sentences for training."""
        return [
            "SRVO-063 BZAL alarm occurs during encoder initialization",
            "Replace pulsecoder if mastering calibration fails",
            "Check cable connections for servo motor communication errors",
            "Run RCAL procedure to recalibrate robot joints",
            "Verify brake release during manual axis movement",
            "Check DCS settings for safe position configuration",
            "MOTN-023 indicates motion supervision fault detected",
            "Reset robot controller after clearing fault conditions",
            "Verify teach pendant is properly connected",
            "Check emergency stop circuit functionality",
        ] * 10  # Repeat to get 100 sentences (minimum for reasonable training)

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_train_adapter_minimal(self, temp_models_dir, sample_sentences):
        """Test training with minimal configuration."""
        adapter = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")

        config = DomainConfig(
            domain_id="test",
            domain_name="Test Domain",
            epochs=1,
            batch_size=4,
        )

        result = await adapter.train_adapter(sample_sentences, config)

        assert result.status == AdaptationStatus.COMPLETED
        assert result.num_sentences == len(sample_sentences)
        assert result.epochs_completed == 1
        assert result.model_path is not None
        assert Path(result.model_path).exists()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_encode_after_training(self, temp_models_dir, sample_sentences):
        """Test encoding after training."""
        adapter = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")

        config = DomainConfig(
            domain_id="test",
            domain_name="Test Domain",
            epochs=1,
            batch_size=4,
        )

        # Train
        await adapter.train_adapter(sample_sentences, config)

        # Encode
        result = await adapter.encode(
            ["SRVO-063 alarm detected", "Robot controller fault"],
            domain_id="test",
        )

        assert result.texts_embedded == 2
        assert result.embeddings.shape[0] == 2
        assert result.dimension > 0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_save_and_reload(self, temp_models_dir, sample_sentences):
        """Test saving and reloading model."""
        # Train and save
        adapter1 = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")
        config = DomainConfig(
            domain_id="reload_test",
            domain_name="Reload Test",
            epochs=1,
            batch_size=4,
        )
        await adapter1.train_adapter(sample_sentences, config)

        # Create new adapter and load
        adapter2 = TSDaeAdapter(models_dir=temp_models_dir, device="cpu")
        loaded = await adapter2.load_adapter("reload_test")

        assert loaded is True
        assert "reload_test" in adapter2._loaded_models

        # Verify encoding works
        result = await adapter2.encode("test query", domain_id="reload_test")
        assert result.texts_embedded == 1
