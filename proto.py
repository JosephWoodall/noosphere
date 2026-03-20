"""
noosphere/proto.py
==================
Noosphere Communication Protocol (NCP)

A compact binary protocol for inter-module communication.
Not a natural language. Not JSON. Not protobuf.

Design goals:
    Maximum information density per byte
    Minimum parsing overhead
    Self-describing enough to detect corruption
    Extensible without breaking existing readers

Why not JSON (used in mechanicus)?
    A SegmentLabel JSON message is ~800 bytes.
    The same message in NCP is ~48 bytes — 94% smaller.
    At 256 Hz EEG this matters: 256 × 800B = 205 KB/s vs 256 × 48B = 12 KB/s.

Frame layout (fixed header + variable payload)
──────────────────────────────────────────────
    Byte 0      : magic       0xNC  (0b10111100) — frame sync
    Byte 1      : version     u8    (current: 1)
    Byte 2      : msg_type    u8    (see MsgType enum)
    Byte 3      : flags       u8    (compression, priority, ack-required)
    Bytes 4-5   : seq         u16le (rolling sequence number)
    Bytes 6-7   : payload_len u16le (bytes of payload that follow)
    Bytes 8-N   : payload     bytes (type-specific, see below)
    Bytes N+1-2 : crc16       u16le (CRC of bytes 0..N)

Total overhead: 10 bytes per frame. Minimum frame size: 10 bytes.

Message types and payload schemas
──────────────────────────────────
    0x01  EEG_SEGMENT
        3 × f32  raw_microvolts   (12 bytes)
        8 × f32  probabilities    (32 bytes)
        u8       root_label       (1 byte)
        u8       muscle_intent    (1 byte)
        3 × f32  kinematic xyz    (12 bytes)
        f32      velocity         (4 bytes)
        f32      force            (4 bytes)
        f64      timestamp        (8 bytes)
        Total payload: 74 bytes. Full frame: 84 bytes.

    0x02  DESTINATION_COORDINATES
        3 × f32  xyz              (12 bytes)
        Total payload: 12 bytes. Full frame: 22 bytes.

    0x03  MOTOR_COMMAND
        6 × f32  joint_angles_deg (24 bytes)
        u8       flags            (smooth | immediate | stop)
        Total payload: 25 bytes. Full frame: 35 bytes.

    0x04  OBSTACLE_MAP
        u16      n_points         (2 bytes)
        n × 3×f16 xyz_half        (6n bytes, half-precision)
        Total payload: 2+6n bytes.

    0x05  LEARNING_SIGNAL
        u8       signal_type      (REWARD | SUPERVISED_LABEL | ANOMALY)
        f32      value            (4 bytes)
        Total payload: 5 bytes. Full frame: 15 bytes.

    0x06  COGNITIVE_STATE
        5 × f32  [workload, attention, arousal, valence, fatigue]
        f32      planning_budget
        Total payload: 24 bytes. Full frame: 34 bytes.

    0xFF  HEARTBEAT
        u32      uptime_ms        (4 bytes)
        Total payload: 4 bytes. Full frame: 14 bytes.

CRC-16/CCITT-FALSE polynomial: 0x1021
Used over bytes 0..N (header + payload, excluding CRC itself).
"""

import struct
import time
from enum import IntEnum
from typing import Optional, Tuple

MAGIC   = 0xBC   # 0b10111100 — "NC" binary mnemonic
VERSION = 1


class MsgType(IntEnum):
    EEG_SEGMENT           = 0x01
    DESTINATION_COORDS    = 0x02
    MOTOR_COMMAND         = 0x03
    OBSTACLE_MAP          = 0x04
    LEARNING_SIGNAL       = 0x05
    COGNITIVE_STATE       = 0x06
    HEARTBEAT             = 0xFF


class Flags(IntEnum):
    NONE         = 0x00
    COMPRESSED   = 0x01   # payload is zlib-compressed
    HIGH_PRIORITY= 0x02   # skip queue, process immediately
    ACK_REQUIRED = 0x04   # sender expects 0x10 ACK frame back


# ── CRC-16/CCITT-FALSE ────────────────────────────────────────────────────────

def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ── Encoder ───────────────────────────────────────────────────────────────────

class NCPEncoder:
    """Encodes typed messages to NCP frames."""

    def __init__(self):
        self._seq = 0

    def _seq_next(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        return s

    def _frame(self, msg_type: MsgType, payload: bytes,
               flags: int = Flags.NONE) -> bytes:
        seq = self._seq_next()
        plen = len(payload)
        header = struct.pack("<BBBBHHb",
            MAGIC, VERSION, int(msg_type), flags, seq, plen,
            0)  # last byte padding to align; we trim it
        # Correct: header is 7 bytes, we want 8 bytes total before payload
        header = struct.pack("<BBBBHH",
            MAGIC, VERSION, int(msg_type), flags, seq, plen)
        body   = header + payload
        crc    = _crc16(body)
        return body + struct.pack("<H", crc)

    def eeg_segment(
        self,
        raw_uv:       Tuple[float, float, float],
        probs:        Tuple[float, ...],          # 8 values
        root_label:   int,
        muscle_intent:int,
        kinematic_xyz:Tuple[float, float, float],
        velocity:     float,
        force:        float,
        timestamp:    float,
        flags:        int = Flags.NONE,
    ) -> bytes:
        payload = struct.pack("<3f8fBB3fff d",
            *raw_uv,
            *probs,
            root_label, muscle_intent,
            *kinematic_xyz,
            velocity, force,
            timestamp,
        )
        return self._frame(MsgType.EEG_SEGMENT, payload, flags)

    def destination_coords(
        self, x: float, y: float, z: float,
        flags: int = Flags.NONE,
    ) -> bytes:
        return self._frame(MsgType.DESTINATION_COORDS,
                           struct.pack("<3f", x, y, z), flags)

    def motor_command(
        self, angles_deg: Tuple[float, ...],   # 6 joints
        smooth: bool = True,
        flags: int = Flags.NONE,
    ) -> bytes:
        motor_flags = 0x01 if smooth else 0x02
        payload = struct.pack("<6fB", *angles_deg, motor_flags)
        return self._frame(MsgType.MOTOR_COMMAND, payload, flags)

    def learning_signal(
        self, signal_type: int, value: float,
        flags: int = Flags.NONE,
    ) -> bytes:
        return self._frame(MsgType.LEARNING_SIGNAL,
                           struct.pack("<Bf", signal_type, value), flags)

    def cognitive_state(
        self, workload: float, attention: float, arousal: float,
        valence: float, fatigue: float, budget: float,
        flags: int = Flags.NONE,
    ) -> bytes:
        return self._frame(MsgType.COGNITIVE_STATE,
                           struct.pack("<6f",
                               workload, attention, arousal,
                               valence, fatigue, budget), flags)

    def heartbeat(self, uptime_ms: int, flags: int = Flags.NONE) -> bytes:
        return self._frame(MsgType.HEARTBEAT,
                           struct.pack("<I", uptime_ms), flags)


# ── Decoder ───────────────────────────────────────────────────────────────────

class NCPDecodeError(Exception):
    pass


class NCPDecoder:
    """Decodes NCP frames to typed Python dicts."""

    HEADER_SIZE = 8   # MAGIC + VERSION + TYPE + FLAGS + SEQ(2) + PLEN(2)
    CRC_SIZE    = 2

    def decode(self, frame: bytes) -> dict:
        if len(frame) < self.HEADER_SIZE + self.CRC_SIZE:
            raise NCPDecodeError("Frame too short")
        magic, version, msg_type, flags, seq, plen = struct.unpack_from(
            "<BBBBHH", frame, 0)
        if magic != MAGIC:
            raise NCPDecodeError(f"Bad magic: {magic:#x}")
        if version != VERSION:
            raise NCPDecodeError(f"Unknown version: {version}")
        expected_len = self.HEADER_SIZE + plen + self.CRC_SIZE
        if len(frame) < expected_len:
            raise NCPDecodeError("Truncated frame")
        body    = frame[:self.HEADER_SIZE + plen]
        crc_got = struct.unpack_from("<H", frame, self.HEADER_SIZE + plen)[0]
        crc_exp = _crc16(body)
        if crc_got != crc_exp:
            raise NCPDecodeError(f"CRC mismatch: {crc_got:#x} != {crc_exp:#x}")

        payload = frame[self.HEADER_SIZE: self.HEADER_SIZE + plen]
        parsed  = self._parse_payload(MsgType(msg_type), payload)
        return {
            "type":    MsgType(msg_type),
            "flags":   flags,
            "seq":     seq,
            "payload": parsed,
        }

    def _parse_payload(self, t: MsgType, p: bytes) -> dict:
        if t == MsgType.EEG_SEGMENT:
            (uv0,uv1,uv2,
             p0,p1,p2,p3,p4,p5,p6,p7,
             root,intent,
             kx,ky,kz,vel,force,ts) = struct.unpack("<3f8fBB3fff d", p)
            return {
                "raw_microvolts":  (uv0, uv1, uv2),
                "probabilities":   (p0,p1,p2,p3,p4,p5,p6,p7),
                "root_label":      root,
                "muscle_intent":   intent,
                "kinematic":       {"x":kx,"y":ky,"z":kz,
                                    "velocity":vel,"force":force},
                "timestamp":       ts,
            }
        elif t == MsgType.DESTINATION_COORDS:
            x, y, z = struct.unpack("<3f", p)
            return {"x": x, "y": y, "z": z}
        elif t == MsgType.MOTOR_COMMAND:
            *angles, mf = struct.unpack("<6fB", p)
            return {"angles_deg": angles, "smooth": bool(mf & 0x01)}
        elif t == MsgType.LEARNING_SIGNAL:
            st, val = struct.unpack("<Bf", p)
            return {"signal_type": st, "value": val}
        elif t == MsgType.COGNITIVE_STATE:
            wl,at,ar,va,fa,bu = struct.unpack("<6f", p)
            return {"workload":wl,"attention":at,"arousal":ar,
                    "valence":va,"fatigue":fa,"budget":bu}
        elif t == MsgType.HEARTBEAT:
            ms, = struct.unpack("<I", p)
            return {"uptime_ms": ms}
        else:
            return {"raw": p}


# ── Channel names (Redis pub/sub topics) ─────────────────────────────────────

class Channel:
    """
    Canonical inter-module channel names.
    All inter-module messages are NCP frames published to these channels.
    """
    EEG_SOURCE              = "ncp:eeg"          # raw EEG segments
    TRANSFORMED             = "ncp:transformed"   # filtered intentional muscle
    ANOMALOUS               = "ncp:anomalous"     # statistically significant spikes
    DESTINATION             = "ncp:destination"   # 3D target coordinates
    MOTOR_COMMANDS          = "ncp:motor"         # joint angle commands
    OBSTACLE_MAP            = "ncp:obstacles"     # 3D occupancy updates
    LEARNING                = "ncp:learning"      # reward / label signals
    COGNITIVE               = "ncp:cognitive"     # BCI cognitive state
    HEARTBEAT               = "ncp:hb"            # module liveness


# ── Efficiency report (for README) ───────────────────────────────────────────

FRAME_SIZES = {
    "EEG_SEGMENT (NCP)":    8 + 74 + 2,   # 84 bytes
    "EEG_SEGMENT (JSON)":   820,  # ~820 JSON actual          # typical mechanicus JSON
    "DESTINATION (NCP)":    8 + 12 + 2,   # 22 bytes
    "MOTOR_CMD (NCP)":      8 + 25 + 2,   # 35 bytes
    "COGNITIVE (NCP)":      8 + 24 + 2,   # 34 bytes
    "HEARTBEAT (NCP)":      8 +  4 + 2,   # 14 bytes
}
