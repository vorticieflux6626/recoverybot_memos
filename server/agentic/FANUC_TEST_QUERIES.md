# FANUC Robot Advanced Troubleshooting Test Queries

> **Updated**: 2025-12-30 | **Parent**: [AGENTIC_OVERVIEW.md](./AGENTIC_OVERVIEW.md) | **Status**: Current Test Suite

This document contains 10 advanced/complex FANUC robot troubleshooting questions designed to test the effectiveness of the agentic search system. These queries are based on real-world problems found on forums like r/PLC, Robot-Forum.com, and FANUC user groups.

## Test Query Set

### 1. Encoder Replacement and RCAL Alarm
**Query:** "M710iC robot SRVO-063 RCAL alarm after replacing pulsecoder on axis 2, single axis mastering fails with RCAL error even after setting $MASTER_DONE flag"

**Complexity:** Expert
**Key Topics:**
- Pulsecoder replacement procedure
- RCAL (Rotation Calibration) alarm resolution
- Single axis mastering vs full mastering
- $MASTER_DONE system variable
- Encoder battery backup requirements

**Why This Tests Well:**
- Requires deep understanding of FANUC calibration system
- Multiple potential causes (hardware, software, procedure)
- Needs cross-referencing of maintenance manual sections

---

### 2. Collision Detection False Triggers
**Query:** "R-2000iC triggers SRVO-050 collision detect alarm during jogging after deadman release, only happens at certain joint configurations near axis 4/5 singularity"

**Complexity:** Hard
**Key Topics:**
- Collision detection tuning ($PARAM_GROUP)
- Singularity behavior in 6-axis robots
- SRVO-050 alarm causes
- Jogging mode sensitivity adjustments
- Axis 4/5 wrist singularity avoidance

**Why This Tests Well:**
- Requires understanding of both hardware and software aspects
- Singularity is a nuanced topic requiring geometric knowledge
- Multiple configuration parameters involved

---

### 3. Servo Amplifier Overheating
**Query:** "SRVO-023 servo amplifier overheat on J4 motor, ambient temp only 25C, robot running 85% cycle time, amplifier replaced but alarm returns after 2 hours of operation"

**Complexity:** Hard
**Key Topics:**
- Servo amplifier thermal management
- Motor sizing and duty cycle calculations
- $MCR parameters for motor current
- Cycle optimization to reduce motor load
- Amplifier vs motor vs cable diagnosis

**Why This Tests Well:**
- Requires thermal/mechanical engineering knowledge
- Could be motor, amplifier, or application issue
- Needs understanding of continuous vs peak torque

---

### 4. EtherNet/IP Communication Failures
**Query:** "R-30iB Plus COMM-010 EtherNet/IP adapter not communicating with ControlLogix PLC after firmware update, ping works but I/O assembly fails, RPI set to 10ms"

**Complexity:** Expert
**Key Topics:**
- EtherNet/IP I/O assembly configuration
- RPI (Requested Packet Interval) settings
- Firmware compatibility between devices
- Connection timeout parameters
- Produced/consumed tag mapping

**Why This Tests Well:**
- Requires knowledge of both FANUC and Rockwell systems
- Network protocol understanding needed
- Multiple configuration points to check

---

### 5. KAREL Program Memory Errors
**Query:** "KAREL program crashes with INTP-127 illegal condition when calling routine with POSITION type parameter, works fine when position is defined locally, using R-30iB controller"

**Complexity:** Expert
**Key Topics:**
- KAREL memory management
- POSITION type handling
- Parameter passing (IN/OUT/IN-OUT)
- Stack memory limitations
- Program structure best practices

**Why This Tests Well:**
- Requires KAREL programming expertise
- Subtle difference between local and parameter passing
- Documentation often incomplete on edge cases

---

### 6. Multi-Axis Mastering After Gearbox Replacement
**Query:** "How to re-master R-2000iC J1-J3 axes after replacing gearbox on J2, need to maintain accuracy for arc welding application with ±0.5mm tolerance, no fixture available"

**Complexity:** Expert
**Key Topics:**
- Multi-axis mastering procedures
- Zero position master vs quick master
- Mastering without fixtures
- $DMR_GRP calibration data
- Welding accuracy requirements

**Why This Tests Well:**
- Complex mechanical procedure
- Multiple valid approaches
- Application-specific accuracy requirements

---

### 7. iRVision Calibration Drift
**Query:** "iRVision 2D localization offset drifts 3-5mm over 8-hour shift, recalibration fixes it temporarily, using fixed mount camera on M-10iA picking from conveyor"

**Complexity:** Hard
**Key Topics:**
- Camera mounting stability
- Thermal expansion effects
- Calibration grid requirements
- Lighting consistency
- Vision process parameters

**Why This Tests Well:**
- Environmental factors involved
- Could be mechanical, optical, or software
- Requires systematic elimination approach

---

### 8. Parameter Corruption After Power Loss
**Query:** "SYST-011 backup checksum error after unexpected power loss, all programs intact but servo parameters corrupted, need to restore from image backup on memory card"

**Complexity:** Moderate
**Key Topics:**
- System backup/restore procedures
- Checksum validation
- $ALL.DT file structure
- Controlled start menu options
- Flash memory card operations

**Why This Tests Well:**
- Critical recovery procedure
- Multiple backup locations possible
- Risk of further data loss

---

### 9. Arc Welding Weave Pattern Issues
**Query:** "Arc welding weave pattern inconsistent at corners of rectangular path, weave seems to compress, using 10mm amplitude triangular weave at 600mm/min travel speed on M-20iA"

**Complexity:** Hard
**Key Topics:**
- Weaving instruction parameters
- Corner blending behavior
- Weave frequency calculations
- TCP speed vs joint speed
- Weld schedule coordination

**Why This Tests Well:**
- Application-specific knowledge required
- Involves motion control and welding interaction
- Geometric path calculations needed

---

### 10. DCS Safety Zone Interference
**Query:** "DCS position check SRVO-230 triggers incorrectly when robot approaches zone boundary, safety zone defined for cell interference but stopping 50mm early, using R-30iB with DCS option"

**Complexity:** Expert
**Key Topics:**
- DCS (Dual Check Safety) configuration
- Safe zone definition methods
- Prediction time and stopping distance
- Tool frame compensation
- Safety-rated vs application zones

**Why This Tests Well:**
- Safety-critical system
- Requires precise understanding of DCS algorithms
- Multiple parameters affect stopping distance

---

## Evaluation Criteria

For each query, evaluate the agentic search system on:

1. **Relevance**: Did it find sources addressing the specific issue?
2. **Accuracy**: Are the suggested solutions technically correct?
3. **Completeness**: Are all major causes/solutions covered?
4. **Source Quality**: Are sources from authoritative FANUC documentation?
5. **Synthesis Quality**: Is the answer well-organized and actionable?
6. **Confidence Score**: Does the confidence level match answer quality?

## Expected Performance

| Query Category | Expected Sources | Min Confidence |
|----------------|------------------|----------------|
| Encoder/Mastering | FANUC manuals, Robot-Forum | 60% |
| Servo Alarms | Studylib, maintenance docs | 55% |
| Communication | Both FANUC and PLC forums | 50% |
| KAREL Programming | KAREL Reference Manual | 45% |
| Vision | iRVision documentation | 50% |
| Safety (DCS) | Safety manual, forums | 55% |
| Arc Welding | Welding-specific docs | 50% |

## Running Tests

```bash
# Test individual query
curl -X POST "http://localhost:8001/api/v1/search/universal/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "<QUERY_TEXT>", "max_iterations": 3, "preset": "research"}'

# Via gateway with classification
curl -X POST "http://localhost:8001/api/v1/search/gateway/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "<QUERY_TEXT>", "preset": "full"}'
```

## Version History

| Date | Change |
|------|--------|
| 2025-12-29 | Initial test query set created for system evaluation |
