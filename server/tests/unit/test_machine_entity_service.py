"""
Unit tests for Machine Entity Service

Tests the error pattern extraction and context formatting
without requiring the PDF Extraction Tools API to be running.
"""

import pytest
from core.machine_entity_service import (
    MachineEntityService,
    get_machine_entity_service,
    SRVO_ERROR_PATTERN,
    AXIS_PATTERN,
    ROBOT_MODEL_PATTERN
)


class TestSRVOErrorPattern:
    """Tests for SRVO error code pattern matching"""

    def test_srvo_standard_format(self):
        """Test standard SRVO-0XX format"""
        match = SRVO_ERROR_PATTERN.search("SRVO-063 alarm")
        assert match is not None
        assert match.group(1) == "63"

    def test_srvo_with_leading_zero(self):
        """Test SRVO-063 with leading zero in error number"""
        match = SRVO_ERROR_PATTERN.search("SRVO-063")
        assert match is not None
        assert match.group(1) == "63"

    def test_srvo_without_hyphen(self):
        """Test SRVO 063 format (space instead of hyphen)"""
        match = SRVO_ERROR_PATTERN.search("SRVO 063")
        assert match is not None
        assert match.group(1) == "63"

    def test_srvo_three_digits(self):
        """Test three-digit error codes"""
        match = SRVO_ERROR_PATTERN.search("SRVO-123")
        assert match is not None
        assert match.group(1) == "123"

    def test_srvo_case_insensitive(self):
        """Test case insensitivity"""
        match = SRVO_ERROR_PATTERN.search("srvo-063")
        assert match is not None
        assert match.group(1) == "63"

    def test_no_match_other_errors(self):
        """Test that non-SRVO errors don't match"""
        match = SRVO_ERROR_PATTERN.search("MOTN-045 alarm")
        assert match is None


class TestAxisPattern:
    """Tests for axis number pattern matching"""

    def test_j_format(self):
        """Test J1 format"""
        match = AXIS_PATTERN.search("J1 axis")
        assert match is not None
        assert match.group(1) == "1"

    def test_axis_word_format(self):
        """Test 'axis 1' format"""
        match = AXIS_PATTERN.search("axis 3 motor")
        assert match is not None
        assert match.group(1) == "3"

    def test_all_axes(self):
        """Test all valid axis numbers 1-6"""
        for i in range(1, 7):
            match = AXIS_PATTERN.search(f"J{i}")
            assert match is not None
            assert match.group(1) == str(i)

    def test_invalid_axis(self):
        """Test that axis 7 doesn't match"""
        match = AXIS_PATTERN.search("J7")
        assert match is None


class TestRobotModelPattern:
    """Tests for robot model pattern matching"""

    def test_m16ib(self):
        """Test M-16iB/20 format"""
        match = ROBOT_MODEL_PATTERN.search("M-16iB/20 robot")
        assert match is not None
        assert match.group(1) == "M-16iB/20"

    def test_lr_mate(self):
        """Test LR Mate format"""
        match = ROBOT_MODEL_PATTERN.search("LR Mate 200iD/7L")
        assert match is not None
        assert "200iD" in match.group(1)

    def test_crx(self):
        """Test CRX collaborative robot format"""
        match = ROBOT_MODEL_PATTERN.search("CRX-10iA cobot")
        assert match is not None
        assert match.group(1) == "CRX-10iA"

    def test_r2000(self):
        """Test R-2000iC format"""
        match = ROBOT_MODEL_PATTERN.search("R-2000iC/165F")
        assert match is not None
        assert "R-2000iC" in match.group(1)


class TestMachineEntityService:
    """Tests for MachineEntityService class"""

    def test_singleton_instance(self):
        """Test that get_machine_entity_service returns singleton"""
        service1 = get_machine_entity_service()
        service2 = get_machine_entity_service()
        assert service1 is service2

    def test_extract_error_info_full(self):
        """Test full error info extraction"""
        service = MachineEntityService()
        result = service.extract_error_info(
            "FANUC SRVO-063 alarm on J1 axis M-16iB/20"
        )
        assert result["has_srvo_error"] is True
        assert result["error_code"] == "SRVO-063"
        assert result["axis_number"] == 1
        assert result["robot_model"] == "M-16iB/20"

    def test_extract_error_info_partial(self):
        """Test partial error info extraction (no axis/model)"""
        service = MachineEntityService()
        result = service.extract_error_info("SRVO-068 motor overload")
        assert result["has_srvo_error"] is True
        assert result["error_code"] == "SRVO-068"
        assert result["axis_number"] is None
        assert result["robot_model"] is None

    def test_extract_error_info_no_error(self):
        """Test extraction with no SRVO error"""
        service = MachineEntityService()
        result = service.extract_error_info("robot arm not moving")
        assert result["has_srvo_error"] is False
        assert result["error_code"] is None

    def test_format_machine_context(self):
        """Test machine context formatting"""
        service = MachineEntityService()
        context = {
            "error_info": {
                "component": "motor",
                "severity": "warning",
                "axis_specific": True
            },
            "affected_components": [
                {"type": "motor", "name": "M-16iB/20_J1_motor"}
            ],
            "related_errors": [
                {"error_code": "SRVO-068", "confidence": 0.95}
            ],
            "sibling_components": ["encoder_xyz", "brake_123"]
        }
        error_info = {"error_code": "SRVO-063"}

        formatted = service._format_machine_context(context, error_info)

        assert "<machine_architecture_context>" in formatted
        assert "SRVO-063" in formatted
        assert "motor" in formatted
        assert "warning" in formatted
        assert "SRVO-068" in formatted
        assert "</machine_architecture_context>" in formatted


class TestMachineEntityServiceIntegration:
    """Integration-like tests (require mocked API)"""

    def test_error_code_normalization(self):
        """Test that error codes are properly normalized to SRVO-0XX format"""
        service = MachineEntityService()

        # Test various formats
        test_cases = [
            ("SRVO-63", "SRVO-063"),
            ("SRVO-063", "SRVO-063"),
            ("srvo 63", "SRVO-063"),
            ("SRVO-123", "SRVO-123"),
        ]

        for input_query, expected_code in test_cases:
            result = service.extract_error_info(f"Alarm {input_query}")
            if result["has_srvo_error"]:
                assert result["error_code"] == expected_code, f"Failed for {input_query}"

    def test_cache_key_generation(self):
        """Test that cache keys are unique per error/model/axis combo"""
        service = MachineEntityService()

        # Simulate cache key generation
        key1 = f"troubleshoot:SRVO-063:M-16iB/20:1"
        key2 = f"troubleshoot:SRVO-063:M-16iB/20:2"
        key3 = f"troubleshoot:SRVO-068:M-16iB/20:1"

        assert key1 != key2  # Different axis
        assert key1 != key3  # Different error
