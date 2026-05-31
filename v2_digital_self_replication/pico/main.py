"""
Raspberry Pi Pico — 28BYJ-48 stepper controller.

Implements the same serial protocol as arduino_bridge.py expects:
  Command  (host → Pico):  "M<idx> <pwm>\\n"   pwm 0–255
  Feedback (Pico → host):  "P<idx> <angle>\\n"  angle 0–180, every 50 ms

Wiring (28BYJ-48 + ULN2003 → Pico):
  ULN2003 VCC → VBUS (pin 40, 5V)
  ULN2003 GND → GND  (pin 38)
  IN1          → GP8  (pin 11)
  IN2          → GP9  (pin 12)
  IN3          → GP10 (pin 14)
  IN4          → GP11 (pin 15)

Upload:
  mpremote cp stepper.py :stepper.py
  mpremote cp main.py    :main.py
"""

import sys
import time
import uselect
from machine import Pin

from stepper import StepperSim

PINS = [
    Pin(8,  Pin.OUT),
    Pin(9,  Pin.OUT),
    Pin(10, Pin.OUT),
    Pin(11, Pin.OUT),
]

STEP_DELAY  = 2000   # microseconds per half-step
WATCHDOG_MS = 500
REPORT_MS   = 50

sim = StepperSim(max_steps=512)


def _apply_phase() -> None:
    phase = sim.phase()
    for i, pin in enumerate(PINS):
        pin.value(phase[i])


def _release() -> None:
    for pin in PINS:
        pin.value(0)


poll = uselect.poll()
poll.register(sys.stdin, uselect.POLLIN)

last_cmd_ms  = time.ticks_ms()
last_report  = time.ticks_ms()
buf          = ""

while True:
    # Non-blocking serial read
    if poll.poll(0):
        ch = sys.stdin.read(1)
        if ch == '\n':
            line = buf.strip()
            buf  = ""
            if line.startswith('M'):
                parts = line[1:].split()
                if len(parts) == 2:
                    try:
                        motor_idx = int(parts[0])
                        pwm       = max(0, min(255, int(parts[1])))
                        if motor_idx == 0:
                            sim.set_pwm(pwm)
                            last_cmd_ms = time.ticks_ms()
                    except ValueError:
                        pass
        else:
            buf += ch

    # Watchdog
    if time.ticks_diff(time.ticks_ms(), last_cmd_ms) > WATCHDOG_MS:
        sim.centre()

    # Step toward target
    if sim.tick():
        _apply_phase()
        time.sleep_us(STEP_DELAY)
    else:
        _release()

    # Report every 50 ms
    now = time.ticks_ms()
    if time.ticks_diff(now, last_report) >= REPORT_MS:
        print(f"P0 {sim.angle()}")
        last_report = now
