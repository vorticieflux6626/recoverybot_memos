"""
Unit Tests for Entity Normalization and Extraction.

Tests the FANUC schema patterns, entity extraction, and normalization
functions used throughout the agentic search system.
"""
import pytest
from typing import List


# ============================================================================
# FANUC Schema Tests
# ============================================================================

class TestFANUCErrorPatterns:
    """Tests for FANUC error code pattern matching."""

    def test_import_fanuc_schema(self):
        """Should import FANUC schema module successfully."""
        from agentic.schemas.fanuc_schema import (
            FANUC_ERROR_PATTERNS,
            FANUC_SCHEMA,
            FANUCEntityType,
            FANUCRelationType,
        )
        assert len(FANUC_ERROR_PATTERNS) > 0
        assert FANUC_SCHEMA is not None
        assert FANUCEntityType.ERROR_CODE == "error_code"
        assert FANUCRelationType.CAUSES == "causes"

    @pytest.mark.parametrize("code,expected", [
        ("SRVO-001", True),
        ("SRVO-063", True),
        ("SRVO-1234", True),
        ("MOTN-023", True),
        ("MOTN-100", True),
        ("SYST-032", True),
        ("HOST-001", True),
        ("INTP-127", True),
        ("PRIO-001", True),
        ("COMM-001", True),
        ("VISI-001", True),
        ("SRIO-001", True),
        ("FILE-001", True),
        ("MACR-001", True),
        ("PALL-001", True),
        ("SPOT-001", True),
        ("ARC-001", True),
        ("DISP-001", True),
        ("INVALID", False),
        ("SERVO-001", False),  # Wrong prefix
        ("SRVO001", False),    # Missing hyphen
        ("SRVO-1", False),     # Too few digits
    ])
    def test_error_code_patterns(self, code: str, expected: bool):
        """Error code patterns should match valid codes."""
        from agentic.schemas.fanuc_schema import COMPILED_ERROR_PATTERNS
        import re

        matched = any(p.search(code) for p in COMPILED_ERROR_PATTERNS)
        assert matched == expected, f"Pattern match for '{code}' should be {expected}"

    def test_extract_error_codes_single(self):
        """Should extract single error code from text."""
        from agentic.schemas.fanuc_schema import extract_error_codes

        text = "The robot shows SRVO-063 alarm on axis J3"
        codes = extract_error_codes(text)
        assert "SRVO-063" in codes

    def test_extract_error_codes_multiple(self):
        """Should extract multiple error codes from text."""
        from agentic.schemas.fanuc_schema import extract_error_codes

        text = "First SRVO-001 appeared, then MOTN-023, finally SYST-100"
        codes = extract_error_codes(text)
        assert len(codes) == 3
        assert "SRVO-001" in codes
        assert "MOTN-023" in codes
        assert "SYST-100" in codes

    def test_extract_error_codes_case_insensitive(self):
        """Should extract error codes regardless of case."""
        from agentic.schemas.fanuc_schema import extract_error_codes

        text = "Found srvo-063 and SRVO-064 alarms"
        codes = extract_error_codes(text)
        assert len(codes) >= 2

    def test_extract_error_codes_empty_text(self):
        """Should return empty list for text with no codes."""
        from agentic.schemas.fanuc_schema import extract_error_codes

        codes = extract_error_codes("No error codes here")
        assert codes == []

    def test_extract_error_codes_deduplication(self):
        """Should deduplicate repeated error codes."""
        from agentic.schemas.fanuc_schema import extract_error_codes

        text = "SRVO-063 appeared twice: SRVO-063 and SRVO-063"
        codes = extract_error_codes(text)
        assert len(codes) == 1
        assert "SRVO-063" in codes


class TestFANUCErrorCategories:
    """Tests for error code category detection."""

    @pytest.mark.parametrize("code,category", [
        ("SRVO-001", "Servo Alarms"),
        ("MOTN-023", "Motion Alarms"),
        ("SYST-100", "System Alarms"),
        ("HOST-001", "Host Communication Alarms"),
        ("INTP-127", "Interpreter Alarms"),
        ("PRIO-001", "Priority Alarms"),
        ("COMM-001", "Communication Alarms"),
        ("VISI-001", "Vision Alarms"),
        ("SRIO-001", "Serial I/O Alarms"),
        ("FILE-001", "File System Alarms"),
        ("MACR-001", "Macro Alarms"),
        ("PALL-001", "Palletizing Alarms"),
        ("SPOT-001", "Spot Welding Alarms"),
        ("ARC-001", "Arc Welding Alarms"),
        ("DISP-001", "Dispense Alarms"),
    ])
    def test_get_error_category(self, code: str, category: str):
        """Should return correct category for error codes."""
        from agentic.schemas.fanuc_schema import get_error_category

        result = get_error_category(code)
        assert result == category

    def test_get_error_category_unknown(self):
        """Should return 'Unknown Category' for unrecognized codes."""
        from agentic.schemas.fanuc_schema import get_error_category

        result = get_error_category("XXXX-001")
        assert result == "Unknown Category"

    def test_get_error_category_no_hyphen(self):
        """Should handle codes without hyphen gracefully."""
        from agentic.schemas.fanuc_schema import get_error_category

        result = get_error_category("SRVO001")
        assert result == "Unknown Category"


class TestFANUCQueryDetection:
    """Tests for FANUC query detection."""

    @pytest.mark.parametrize("query,expected", [
        # Error codes
        ("What does SRVO-063 mean?", True),
        ("How to fix MOTN-023 alarm", True),
        ("SYST-100 troubleshooting", True),
        # Robot models
        ("R-2000iC/165F payload", True),
        ("M-710iC/50 specifications", True),
        ("LR Mate 200iD programming", True),
        # Keywords
        ("fanuc robot maintenance", True),
        ("teach pendant not responding", True),
        ("pulsecoder replacement", True),
        ("mastering procedure", True),
        ("RCAL calibration", True),
        ("KAREL programming", True),
        ("TP program error", True),
        ("roboguide simulation", True),
        # Non-FANUC queries
        ("how to cook pasta", False),
        ("python programming", False),
        ("ABB robot errors", False),
        ("KUKA robot alarms", False),
    ])
    def test_is_fanuc_query(self, query: str, expected: bool):
        """Should correctly detect FANUC-related queries."""
        from agentic.schemas.fanuc_schema import is_fanuc_query

        result = is_fanuc_query(query)
        assert result == expected, f"Query '{query}' should be FANUC={expected}"


class TestFANUCModelPatterns:
    """Tests for FANUC robot model pattern matching."""

    @pytest.mark.parametrize("model", [
        "R-2000iC/165F",
        "R-2000iC",
        "M-710iC/50",
        "M-20iA",
        "LR Mate 200iD/7L",
        "LR Mate 200iD",
        "Arc Mate 100iC",
        "CR-35iA",
        "CRX-10iA/L",
    ])
    def test_model_patterns_match(self, model: str):
        """Should match valid robot model patterns."""
        from agentic.schemas.fanuc_schema import COMPILED_MODEL_PATTERNS

        matched = any(p.search(model) for p in COMPILED_MODEL_PATTERNS)
        assert matched, f"Model '{model}' should match pattern"

    def test_model_patterns_no_match_invalid(self):
        """Should not match invalid model strings."""
        from agentic.schemas.fanuc_schema import COMPILED_MODEL_PATTERNS

        invalid_models = ["KUKA KR-210", "ABB IRB-6700", "ROBOT-123"]
        for model in invalid_models:
            matched = any(p.search(model) for p in COMPILED_MODEL_PATTERNS)
            assert not matched, f"Invalid model '{model}' should not match"


class TestFANUCDomainSchema:
    """Tests for FANUC domain schema structure."""

    def test_create_schema(self):
        """Should create schema with all entity types."""
        from agentic.schemas.fanuc_schema import create_fanuc_domain_schema

        schema = create_fanuc_domain_schema()
        assert schema.name == "fanuc_robotics"
        assert len(schema.entities) == 11  # 11 entity types
        assert len(schema.relationships) > 0

    def test_schema_entity_types(self):
        """Schema should include all defined entity types."""
        from agentic.schemas.fanuc_schema import create_fanuc_domain_schema

        schema = create_fanuc_domain_schema()
        entity_names = {e.name for e in schema.entities}

        expected_types = {
            "error_code", "robot_model", "controller", "parameter",
            "io_signal", "register", "component", "procedure",
            "measurement", "part_number", "safety"
        }

        assert entity_names == expected_types

    def test_schema_entity_patterns(self):
        """Each entity should have patterns or keywords."""
        from agentic.schemas.fanuc_schema import create_fanuc_domain_schema

        schema = create_fanuc_domain_schema()
        for entity in schema.entities:
            has_patterns = len(entity.patterns) > 0
            has_keywords = len(entity.keywords) > 0
            assert has_patterns or has_keywords, \
                f"Entity '{entity.name}' should have patterns or keywords"

    def test_schema_relationships(self):
        """Schema should have valid relationship tuples."""
        from agentic.schemas.fanuc_schema import create_fanuc_domain_schema

        schema = create_fanuc_domain_schema()
        for rel in schema.relationships:
            assert len(rel) == 3
            source, rel_type, target = rel
            assert isinstance(source, str)
            assert isinstance(rel_type, str)
            assert isinstance(target, str)

    def test_singleton_schema(self):
        """FANUC_SCHEMA should be a singleton instance."""
        from agentic.schemas.fanuc_schema import FANUC_SCHEMA, create_fanuc_domain_schema

        new_schema = create_fanuc_domain_schema()
        assert FANUC_SCHEMA.name == new_schema.name
        assert len(FANUC_SCHEMA.entities) == len(new_schema.entities)


class TestFANUCComponentPatterns:
    """Tests for FANUC component pattern matching."""

    @pytest.mark.parametrize("text,expected_component", [
        ("J1 motor needs replacement", "axis"),
        ("axis 3 is overheating", "axis"),
        ("servo motor failure", "servo"),
        ("servo amplifier fault", "servo"),
        ("teach pendant display error", "teach_pendant"),
        ("pulsecoder malfunction", "encoder"),
        ("encoder signal lost", "encoder"),
        ("brake release procedure", "brake"),
        ("gearbox noise issue", "gearbox"),
        ("cable assembly damaged", "cable"),
    ])
    def test_component_pattern_matching(self, text: str, expected_component: str):
        """Should extract component type from text."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_COMPONENT_PATTERNS

        found_component = None
        for pattern, component_type in FANUC_COMPONENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found_component = component_type
                break

        assert found_component == expected_component, \
            f"Text '{text}' should match component '{expected_component}'"


class TestFANUCIOPatterns:
    """Tests for I/O signal pattern matching."""

    @pytest.mark.parametrize("signal", [
        "DI[1]", "DO[101]", "RI[5]", "RO[10]",
        "UI[1]", "UO[1]", "SI[1]", "SO[1]",
        "WI[1]", "WO[1]", "GI[1]", "GO[1]",
        "AI[1]", "AO[1]",
    ])
    def test_io_signal_patterns(self, signal: str):
        """Should match valid I/O signal patterns."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_IO_PATTERNS

        matched = any(re.search(p, signal) for p in FANUC_IO_PATTERNS)
        assert matched, f"I/O signal '{signal}' should match pattern"


class TestFANUCRegisterPatterns:
    """Tests for register pattern matching."""

    @pytest.mark.parametrize("register", [
        "R[1]", "R[100]",
        "PR[1]", "PR[50]",
        "SR[1]", "SR[10]",
        "AR[1]", "VR[1]",
    ])
    def test_register_patterns(self, register: str):
        """Should match valid register patterns."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_REGISTER_PATTERNS

        matched = any(re.search(p, register) for p in FANUC_REGISTER_PATTERNS)
        assert matched, f"Register '{register}' should match pattern"


class TestFANUCParameterPatterns:
    """Tests for parameter pattern matching."""

    @pytest.mark.parametrize("param", [
        "$PARAM_GROUP",
        "$MCR[1].STATUS",
        "$SCR_GRP[1]",
        "$MOTYPE",
        "$SPEED",
        "$TERMTYPE",
        "$MASTER",
        "$DCSS_CFG",
    ])
    def test_parameter_patterns(self, param: str):
        """Should match valid parameter patterns."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_PARAMETER_PATTERNS

        matched = any(re.search(p, param) for p in FANUC_PARAMETER_PATTERNS)
        assert matched, f"Parameter '{param}' should match pattern"


class TestFANUCMeasurementPatterns:
    """Tests for measurement pattern matching."""

    @pytest.mark.parametrize("measurement", [
        "100mm", "50.5mm",
        "10cm", "2.5m",
        "45deg", "90°",
        "1500rpm",
        "24V", "200A",
        "5.5kW",
        "50Nm", "100N·m",
        "25kg", "50lb",
        "100ms", "5sec",
        "60Hz",
        "75%",
    ])
    def test_measurement_patterns(self, measurement: str):
        """Should match valid measurement patterns."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_MEASUREMENT_PATTERNS

        matched = any(re.search(p, measurement, re.IGNORECASE) for p in FANUC_MEASUREMENT_PATTERNS)
        assert matched, f"Measurement '{measurement}' should match pattern"


class TestFANUCPartNumberPatterns:
    """Tests for part number pattern matching."""

    @pytest.mark.parametrize("part", [
        "A06B-6079-H101",
        "A06B-0238-B605",
        "A05B-2518-C200",
    ])
    def test_part_number_patterns(self, part: str):
        """Should match valid FANUC part numbers."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_PART_PATTERNS

        matched = any(re.search(p, part) for p in FANUC_PART_PATTERNS)
        assert matched, f"Part number '{part}' should match pattern"


class TestFANUCSafetyPatterns:
    """Tests for safety-related pattern matching."""

    @pytest.mark.parametrize("text", [
        "DCS configuration",
        "Dual Check Safety enabled",
        "SafeMove monitoring",
        "e-stop pressed",
        "emergency stop circuit",
        "deadman switch",
        "safety fence breach",
        "light curtain triggered",
        "safe speed monitoring",
        "safe position check",
        "collaborative mode",
        "cobot operation",
    ])
    def test_safety_patterns(self, text: str):
        """Should match safety-related patterns."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_SAFETY_PATTERNS

        matched = any(re.search(p, text, re.IGNORECASE) for p in FANUC_SAFETY_PATTERNS)
        assert matched, f"Safety text '{text}' should match pattern"


class TestFANUCProcedurePatterns:
    """Tests for procedure pattern matching."""

    @pytest.mark.parametrize("text,expected_procedure", [
        ("mastering procedure", "mastering"),
        ("RCAL calibration", "calibration"),
        ("zero point return", "zero_point"),
        ("backup robot data", "backup"),
        ("restore settings", "restore"),
        ("cold start required", "cold_start"),
        ("controlled start mode", "controlled_start"),
        ("hot start allowed", "hot_start"),
        ("power cycle robot", "power_cycle"),
        ("reset alarm", "reset"),
        ("clear alarm history", "clear_alarm"),
        ("teaching mode", "teaching"),
        ("jogging operation", "jogging"),
        ("payload identification", "payload_id"),
    ])
    def test_procedure_pattern_matching(self, text: str, expected_procedure: str):
        """Should extract procedure type from text."""
        import re
        from agentic.schemas.fanuc_schema import FANUC_PROCEDURE_PATTERNS

        found_procedure = None
        for pattern, procedure_type in FANUC_PROCEDURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found_procedure = procedure_type
                break

        assert found_procedure == expected_procedure, \
            f"Text '{text}' should match procedure '{expected_procedure}'"


# ============================================================================
# Entity Type Enum Tests
# ============================================================================

class TestFANUCEnums:
    """Tests for FANUC enumeration types."""

    def test_entity_type_values(self):
        """EntityType enum should have correct values."""
        from agentic.schemas.fanuc_schema import FANUCEntityType

        assert FANUCEntityType.ERROR_CODE.value == "error_code"
        assert FANUCEntityType.ROBOT_MODEL.value == "robot_model"
        assert FANUCEntityType.CONTROLLER.value == "controller"
        assert FANUCEntityType.COMPONENT.value == "component"

    def test_relation_type_values(self):
        """RelationType enum should have correct values."""
        from agentic.schemas.fanuc_schema import FANUCRelationType

        assert FANUCRelationType.CAUSES.value == "causes"
        assert FANUCRelationType.RESOLVED_BY.value == "resolved_by"
        assert FANUCRelationType.TRIGGERS.value == "triggers"

    def test_entity_types_are_strings(self):
        """Entity types should be string enums."""
        from agentic.schemas.fanuc_schema import FANUCEntityType

        for entity_type in FANUCEntityType:
            assert isinstance(entity_type.value, str)

    def test_relation_types_are_strings(self):
        """Relation types should be string enums."""
        from agentic.schemas.fanuc_schema import FANUCRelationType

        for rel_type in FANUCRelationType:
            assert isinstance(rel_type.value, str)
