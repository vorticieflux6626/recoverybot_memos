#!/usr/bin/env python3
"""
Complex Synthesis Contexts - Challenging scenarios to differentiate model capabilities.

These contexts require:
- Multi-step reasoning chains
- Cross-domain knowledge integration
- Technical specification interpretation
- Causal analysis and root cause identification
- Procedure synthesis from scattered information
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
import re
from datetime import datetime, timezone
from pathlib import Path

# Complex multi-domain scenarios requiring sophisticated reasoning
COMPLEX_CONTEXTS = {
    "cascading_failure_analysis": {
        "name": "Cascading System Failure Analysis",
        "query": """A plastics manufacturing cell experienced a cascading failure sequence:
1. Chiller compressor tripped on high head pressure (410 psi, limit 400)
2. 45 seconds later, injection molding machine barrel zone 4 alarmed for over-temperature (+12°C above setpoint)
3. 2 minutes later, robot SRVO-006 hand broken alarm triggered during part extraction
4. Parts from that cycle showed sink marks and flash simultaneously

Analyze the causal chain. What was the root cause and what protective measures failed? The chiller was running R-410A refrigerant, ambient was 95°F, and the cooling tower had been serviced 2 weeks prior.""",
        "context": """
### Industrial Chiller Troubleshooting Guide

High head pressure in chillers indicates the condenser cannot reject heat effectively. Common causes:
- Dirty condenser coils (most common - 60% of cases)
- Overcharge of refrigerant
- Non-condensables (air) in system
- Ambient temperature exceeding design limits
- Cooling tower issues (scale, fan failure, low water flow)

R-410A operating pressures:
- Normal head pressure: 300-350 psi at 95°F ambient
- High pressure cutout typically set at 400-425 psi
- Subcooling should be 10-15°F

### Injection Molding Temperature Control

Barrel zones are cooled by circulating water/oil. Zone 4 is typically near the feed throat.
Temperature rise sequence during cooling loss:
- First 30 seconds: PID increases heater duty cycle
- 30-60 seconds: Temperature begins rising as heat cannot dissipate
- 60-120 seconds: Over-temperature alarm triggers (typically +10°C above setpoint)

Melt temperature affects:
- Too hot: Flash (material too fluid, enters parting line)
- Cooling too fast: Sink marks (insufficient pack pressure as material shrinks)
- Simultaneous sink marks AND flash indicates process out of control

### FANUC Robot Servo Alarms

SRVO-006: Hand broken - detected via motor current monitoring
- Triggers when current exceeds threshold during motion
- Common causes: Collision, stuck gripper, part interference
- Can be triggered by warped or stuck parts

Robot motion during thermal events:
- Hot parts may deform, making extraction difficult
- Part may stick to mold due to inadequate cooling
- Gripper may fail to seat properly on malformed part

### Cooling Tower Maintenance Effects

After service, common issues:
- Improper fan belt tension
- Water treatment chemistry imbalance
- Clogged spray nozzles
- Incorrect water level setpoint
- Scale buildup from treatment chemicals

Impact timeline:
- Immediate issues: Fan or pump problems
- 1-2 weeks: Chemistry issues manifest
- 2-4 weeks: Scale buildup significant
""",
        "expected_analysis": [
            "cooling tower service caused scale/chemistry issue",
            "high ambient + compromised tower = chiller overload",
            "lost cooling caused melt temperature rise",
            "hot material caused flash + poor cooling caused sink marks",
            "deformed part caused robot extraction collision"
        ],
        "difficulty": "expert",
        "reasoning_type": "causal_chain"
    },

    "plc_network_timing": {
        "name": "PLC Network Timing Analysis",
        "query": """An Allen-Bradley ControlLogix system controlling 6 robot cells via EtherNet/IP is experiencing sporadic faults. The symptoms:
- Random SRVO-050 (collision detection) alarms on different robots, not correlated with actual collisions
- PLC scan time varies from 8ms to 45ms (normally steady at 10ms)
- Network switch shows no errors, link lights stable
- Faults occur more frequently during shift changes

The 1756-EN2T module is configured as the network adapter. IO tree shows 150 devices. RPI is set to 10ms for robot IO. The switch is a Stratix 5700 managed switch.

What is causing the false collision alarms and what's the fix?""",
        "context": """
### EtherNet/IP Network Design Guidelines

Recommended network architecture:
- Maximum 50-60 devices per EN2T adapter
- RPI (Requested Packet Interval) should be 2x PLC scan time minimum
- Total IO packet size affects network loading
- Multicast traffic can cause switch flooding

1756-EN2T specifications:
- Maximum connections: 256
- Recommended: Keep below 128 for stability
- Each robot typically uses 2-4 connections
- Produced/consumed tags add connection overhead

Scan time variability causes:
- Network retries add to scan time
- Large program tasks blocking
- Excessive IO tree updates
- Memory fragmentation over time

### FANUC EtherNet/IP Configuration

Robot EtherNet/IP adapter expects consistent IO updates:
- Missed updates can trigger fault
- SRVO-050 uses torque monitoring - affected by control loop timing
- If position command jitters, servo detects as resistance
- Collision sensitivity register R[147] default = 100%

Network timing requirements:
- IO refresh must be faster than robot interpolation period (typically 8ms)
- Jitter > 2ms can affect servo performance
- Connection timeout typically 4x RPI

### Stratix Switch Configuration

Default settings may not be optimal for industrial control:
- Spanning tree can cause 30-second delays on topology change
- IGMP snooping should be enabled for multicast efficiency
- Port fast should be enabled on end device ports
- QoS should prioritize EtherNet/IP traffic (DSCP 55)

Shift change network effects:
- HMIs reconnecting cause connection storms
- Badge readers on same network cause traffic spikes
- IT traffic from shift reports can consume bandwidth

### ControlLogix Performance Tuning

Reducing scan time variation:
- Separate continuous tasks from event tasks
- Use periodic tasks for IO instead of continuous
- Optimize program structure (avoid AOI nesting)
- Monitor connection statistics in EN2T
""",
        "expected_analysis": [
            "150 devices exceeds recommended 50-60 per adapter",
            "RPI 10ms matches scan time - should be 2x",
            "shift change causes HMI reconnection storms",
            "network jitter causes servo timing issues",
            "false collision from position command jitter"
        ],
        "difficulty": "expert",
        "reasoning_type": "systematic_diagnosis"
    },

    "thermal_expansion_compensation": {
        "name": "Multi-System Thermal Compensation",
        "query": """A precision assembly cell has accuracy problems that worsen through the day:
- Morning: Robot picks parts ±0.1mm accuracy (spec)
- Afternoon: Accuracy degrades to ±0.5mm, vision faults increase
- Night shift (cooler): Accuracy returns to spec

Equipment: FANUC M-10iA robot, Cognex In-Sight 7000 camera, granite surface plate fixture, steel parts
Cell temperature: 68°F morning, 85°F afternoon
Robot mastered at 72°F

Calculate the expected thermal expansion effects and recommend compensation strategy.""",
        "context": """
### Thermal Expansion Coefficients

Material expansion rates (per °F):
- Steel: 6.5 × 10⁻⁶ in/in/°F
- Aluminum: 13 × 10⁻⁶ in/in/°F
- Granite: 4.5 × 10⁻⁶ in/in/°F
- Cast iron (robot base): 5.5 × 10⁻⁶ in/in/°F

Robot arm expansion calculation:
- Total reach affects TCP position
- Each axis bearing has expansion
- Reducer backlash changes with temperature
- Servo motor position may drift

### FANUC Thermal Compensation

Built-in compensation features:
- Cell calibration routine stores reference temperature
- $ROBOT_CAL_TEMP stores mastering temperature
- Automatic compensation only for motor heating, not ambient

Manual compensation methods:
- User frame offset based on temperature sensor
- Macro program to adjust PR[] based on analog input
- Vision-guided offset correction per pick

Typical robot thermal drift:
- 0.02mm per °C at TCP for medium robots
- 17°F change = 9.4°C change
- Expected drift: ~0.19mm from robot alone

### Vision System Thermal Effects

Camera mounting expansion:
- Aluminum bracket expands significantly
- Changes camera angle and working distance
- 500mm aluminum bracket at ΔT=17°F: 0.11mm length change
- Angular change affects calibration

Lens thermal effects:
- Focal length changes with temperature
- Image scale changes 0.001%/°C typical
- At 500mm FOV: 0.05mm scale change per 10°C

Calibration drift:
- Vision calibration valid within ±5°F of cal temperature
- Outdoor lighting through windows changes afternoon exposure

### Granite Surface Plate

Thermal stability:
- Granite is most stable common material
- 48" × 36" plate: 17°F rise = 0.037" (0.9mm) diagonal expansion
- Parts fixtured on plate move with expansion
- Steel parts on granite have differential expansion

Calculation for steel part on granite:
- 12" steel part: grows 0.0013" (0.033mm)
- Granite below part: grows 0.0009" (0.023mm)
- Net datum shift: 0.01mm per 12" of part length
""",
        "expected_analysis": [
            "17°F temp swing causes multi-factor expansion",
            "robot drift ~0.19mm from thermal expansion",
            "camera bracket/lens adds additional error",
            "granite plate + part differential expansion",
            "combined errors exceed 0.4mm tolerance",
            "need temperature compensation macro or climate control"
        ],
        "difficulty": "expert",
        "reasoning_type": "quantitative_analysis"
    },

    "servo_tuning_diagnosis": {
        "name": "Servo System Resonance Diagnosis",
        "query": """A FANUC robot exhibits vibration and following error during high-speed motion:
- Following error alarm SRVO-024 at >80% speed
- Audible 45Hz vibration from J2 axis
- Vibration worse with 15kg payload vs 5kg payload
- New robot, same program worked fine on old unit
- Reducer and motor replaced under warranty, problem persists

Old robot serial: R-30iA (2015), New robot: R-30iB Plus (2024)
Same end-effector, same fixtures, same program.

Diagnose the resonance source and recommend tuning approach.""",
        "context": """
### Servo System Resonance Theory

Resonance occurs when excitation frequency matches natural frequency:
- Mechanical resonance: Structure vibrates at natural frequency
- Servo resonance: Control loop amplifies oscillation
- Anti-resonance: Frequencies where system absorbs energy

Natural frequency calculation:
f = (1/2π) × √(k/m)
- k = stiffness (N/m)
- m = mass (kg)
- Higher mass = lower frequency
- Higher stiffness = higher frequency

### FANUC Servo Parameters

Key tuning parameters:
- Position gain (Kp): Higher = faster response, more oscillation risk
- Velocity gain (Kv): Controls damping
- Acceleration feedforward: Reduces following error
- Notch filter: Attenuates specific frequencies

Default parameter differences R-30iA vs R-30iB Plus:
- R-30iB Plus has higher default gains for faster cycle times
- Compliance control differs
- Torque limiting behavior changed
- New disturbance observer algorithm

Servo oscillation diagnosis:
- Use oscilloscope function in teach pendant
- Check position deviation waveform
- FFT analysis shows resonance frequency
- Frequency shifts with payload indicates mechanical resonance

### Mechanical Resonance Sources

Common resonance sources in robot cells:
- End effector compliance
- Base mounting stiffness
- Cable carrier (dress pack) mass
- Payload mounting stiffness

45Hz resonance typical causes:
- Flexible tool mounting
- Robot base on inadequate foundation
- Harmonic drive compliance with heavy payload

Payload effect on resonance:
- Added mass lowers natural frequency
- 3:1 payload ratio → √3:1 frequency ratio
- 15kg vs 5kg: frequency drops by factor of 1.73
- May shift resonance into servo bandwidth

### Tuning Strategies

Mechanical solutions:
- Stiffen end effector mounting
- Add mass damper
- Reinforce robot base
- Reduce cable carrier slack

Servo solutions:
- Enable notch filter at resonance frequency
- Reduce position gain
- Adjust acceleration profile
- Enable vibration suppression (R-30iB Plus feature)

Specific R-30iB Plus settings:
- $PARAM_GROUP[1].$NOTCH_FREQ = resonance frequency
- $PARAM_GROUP[1].$NOTCH_WIDTH = bandwidth (typically 5-10Hz)
- $MCR.$GENOVERRIDE = reduce max speed
- Disturbance observer gain adjustment
""",
        "expected_analysis": [
            "R-30iB Plus higher default gains causing instability",
            "45Hz resonance shifts with payload = mechanical resonance",
            "heavier payload lowers frequency into servo bandwidth",
            "end effector or mounting compliance likely source",
            "configure notch filter at 45Hz",
            "may need to reduce position gain from defaults"
        ],
        "difficulty": "expert",
        "reasoning_type": "root_cause_analysis"
    },

    "distributed_io_failure": {
        "name": "Distributed IO Intermittent Failure",
        "query": """A Siemens S7-1500 with ET200SP distributed IO racks has intermittent failures:
- 3 of 8 IO racks randomly go offline for 2-5 seconds then recover
- Happens 5-10 times per shift, no pattern to which racks
- Started 2 weeks after a "minor" network change
- Network diagnostics show BF (bus fault) LED flashing during events
- Profinet cable tester shows all cables pass

Network: Linear topology, 8 ET200SP racks, 100m total cable length
Change made: Added a new HMI panel to the network

What is causing the intermittent failures?""",
        "context": """
### Profinet Network Topology

Topology types and reliability:
- Star: Most reliable, switch isolates faults
- Line/Linear: Single point of failure, but simple
- Ring: Redundancy, requires managed switches

ET200SP in linear topology:
- Each rack is a repeater
- Signal regenerated at each device
- Cumulative delay adds up
- Single cable fault affects downstream

Device limits:
- Max 32 devices in line topology
- Max cascaded device delay: 32 × 0.5µs = 16µs
- Total cable length limit: 100m per segment

### Profinet Diagnostics

BF (Bus Fault) LED meanings:
- Solid: No network connection
- Flashing: Communication interrupted/recovering
- 2-5 second recovery typical of:
  - Network topology change detection
  - Name resolution conflict
  - IP address conflict
  - Duplicate device name

Diagnostic tools:
- S7-1500 web server → Diagnostics → Profinet
- Look for "Communication error" events
- Check "Neighbor detection" status
- Examine port statistics for CRC errors

### Common Profinet Issues

After network changes:
- Spanning tree reconvergence (if using standard switches)
- IP conflict if DHCP used
- Duplicate device name (common HMI issue!)
- ARP table conflicts
- LLDP neighbor change causes brief interrupt

HMI on Profinet network issues:
- HMIs often configured with same default device name
- May request same IP range
- HMI polling can load network
- Windows network stack sends unexpected broadcasts

Device name considerations:
- Each Profinet device needs unique name
- Name is case-insensitive
- HMI panels often have factory default names
- Name conflict causes both devices to fault

### ET200SP Recovery Behavior

When connection lost:
- Outputs go to safe state (configurable)
- IO controller attempts reconnection
- Automatic recovery when communication restored
- Recovery time depends on watchdog settings

Watchdog configuration:
- Default watchdog: 2-10 seconds typical
- Faster watchdog = faster fault detection
- But too fast = nuisance faults from jitter

Linear topology cascade effect:
- Upstream fault affects all downstream
- Random rack pattern suggests:
  - Not a single cable fault
  - Network-wide event affecting arbitrarily
  - Points to broadcast storm or name conflict
""",
        "expected_analysis": [
            "HMI may have duplicate Profinet device name",
            "new HMI causing periodic network disruption",
            "linear topology means events cascade randomly",
            "BF flashing indicates communication recovering",
            "check HMI device name matches network config",
            "may need to assign unique name to HMI"
        ],
        "difficulty": "advanced",
        "reasoning_type": "systematic_diagnosis"
    },

    "hydraulic_servo_interaction": {
        "name": "Hydraulic-Electric Servo Interaction",
        "query": """An injection molding machine with hybrid hydraulic/electric drive has inconsistent shot size:
- Shot weight varies ±3% (spec: ±0.5%)
- Variation is cyclical - good shots, then bad, then good
- Electric screw drive position is accurate
- Hydraulic injection pressure shows 50-bar oscillation during pack/hold
- Problem started after hydraulic oil change to different brand

Machine: 450-ton hybrid IMM with electric plasticizing, hydraulic injection
Oil changed from Mobil DTE 25 to generic ISO 46 hydraulic oil
Accumulator pre-charge last checked 6 months ago

Analyze the interaction between systems and identify root cause.""",
        "context": """
### Hybrid Injection Molding Hydraulics

Hydraulic injection cylinder control:
- Proportional valve controls flow/pressure
- Accumulator provides peak flow demand
- Oil viscosity affects valve response
- Servo valve requires clean, consistent oil

Pressure oscillation causes:
- Accumulator pre-charge incorrect (most common)
- Proportional valve instability
- Oil viscosity mismatch
- Air in system
- Pump pulsation transmitted

Pack/hold phase requirements:
- Precise pressure control ±2%
- Valve response time critical
- Oil compressibility affects control
- Higher viscosity = slower response

### Hydraulic Oil Properties

Mobil DTE 25:
- ISO VG 46
- VI (viscosity index): 98
- Pour point: -27°C
- Anti-wear additives: zinc-based

Generic ISO 46 may differ:
- VI could be lower (90-95)
- Different additive package
- May not meet servo valve requirements
- Foaming characteristics may differ

Viscosity effects on servo valves:
- Higher viscosity = slower response
- Lower VI = more variation with temperature
- Wrong viscosity causes oscillation
- Servo valves spec specific viscosity range (typically 15-100 cSt)

### Accumulator Function

Bladder accumulator in IMM:
- Pre-charge: typically 70-80% of working pressure
- Provides instantaneous flow for injection
- Dampens pump pulsation
- Pre-charge loss causes pressure oscillation

Pre-charge degradation:
- Nitrogen permeates through bladder over time
- Loss rate: 5-10% per year typical
- Low pre-charge causes:
  - Delayed pressure response
  - Oscillation during transition
  - Pump struggling to maintain pressure

Checking pre-charge:
- Relieve system pressure completely
- Check nitrogen pressure with gauge
- Should be 70-80% of min system pressure
- Top up with dry nitrogen only

### Shot Weight Variation Analysis

Causes of cyclical variation:
- Thermal cycling (but electric screw is accurate)
- Hydraulic system cycling through pressure
- Accumulator bladder bottoming out cyclically
- Check valve leaking intermittently

Relationship pressure oscillation to shot:
- 50-bar oscillation during pack = ~5% variation
- Pack pressure directly affects density
- Oscillation transfers to part weight
- Matches observed ±3% variation

Interaction with electric plasticizing:
- Electric screw provides consistent melt
- Hydraulic injection introduces variation
- Rules out material or screw issues
- Points to hydraulic control problem
""",
        "expected_analysis": [
            "oil change to generic caused viscosity/VI difference",
            "different oil properties affect servo valve response",
            "accumulator may need pre-charge check after 6 months",
            "50-bar oscillation during pack causes shot variation",
            "recommend returning to specified oil grade",
            "check and correct accumulator pre-charge"
        ],
        "difficulty": "expert",
        "reasoning_type": "cross_domain_synthesis"
    }
}


class ComplexContextDB:
    """Database for complex synthesis contexts."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "synthesis_contexts.db"
        self.db_path = str(db_path)

    def store_context(self, context_id: str, data: dict) -> bool:
        """Store a complex context."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT OR REPLACE INTO synthesis_contexts
                (context_id, query, domains, difficulty, preset_used, retrieved_context,
                 source_count, source_urls, retrieval_duration_ms, timestamp, expected_topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                context_id,
                data['query'],
                json.dumps([data.get('reasoning_type', 'complex')]),
                data['difficulty'],
                'expert_crafted',  # Not from web search
                data['context'],
                1,  # Single curated context
                json.dumps([]),
                0,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(data.get('expected_analysis', []))
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing context: {e}")
            return False
        finally:
            conn.close()

    def get_context(self, context_id: str) -> dict:
        """Get a context by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM synthesis_contexts WHERE context_id = ?", (context_id,))
        row = cur.fetchone()
        conn.close()

        return dict(row) if row else None


def seed_complex_contexts():
    """Seed database with complex contexts."""
    db = ComplexContextDB()

    for ctx_id, ctx_data in COMPLEX_CONTEXTS.items():
        full_id = f"complex_{ctx_id}"
        if db.store_context(full_id, ctx_data):
            print(f"✓ Stored: {full_id} ({ctx_data['name']})")
            print(f"  Reasoning type: {ctx_data['reasoning_type']}")
            print(f"  Context size: {len(ctx_data['context'])} chars")
        else:
            print(f"✗ Failed: {full_id}")

    return list(COMPLEX_CONTEXTS.keys())


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Complex synthesis contexts")
    parser.add_argument("--seed", action="store_true", help="Seed contexts into database")
    parser.add_argument("--list", action="store_true", help="List available contexts")
    parser.add_argument("--show", type=str, help="Show specific context")
    args = parser.parse_args()

    if args.list:
        print("Complex synthesis contexts:")
        for ctx_id, ctx_data in COMPLEX_CONTEXTS.items():
            print(f"\n  {ctx_id}:")
            print(f"    Name: {ctx_data['name']}")
            print(f"    Type: {ctx_data['reasoning_type']}")
            print(f"    Difficulty: {ctx_data['difficulty']}")
        return

    if args.show:
        if args.show in COMPLEX_CONTEXTS:
            ctx = COMPLEX_CONTEXTS[args.show]
            print(f"\n{ctx['name']}")
            print("=" * 60)
            print(f"\nQuery:\n{ctx['query']}")
            print(f"\nExpected Analysis Points:")
            for point in ctx.get('expected_analysis', []):
                print(f"  • {point}")
        else:
            print(f"Unknown context: {args.show}")
        return

    if args.seed:
        print("Seeding complex contexts...")
        seed_complex_contexts()
        return

    # Default: seed
    print("Seeding complex contexts...")
    seed_complex_contexts()


if __name__ == "__main__":
    asyncio.run(main())
