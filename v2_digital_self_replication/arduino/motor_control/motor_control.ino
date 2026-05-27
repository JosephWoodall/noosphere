/*
 * motor_control.ino — Digital Twin Arduino bridge (28BYJ-48 stepper)
 *
 * Driver board: ULN2003
 * Wiring:  IN1→pin8  IN2→pin9  IN3→pin10  IN4→pin11
 *          5V→5V  GND→GND  (motor connector stays on driver board)
 *
 * Command protocol  (bridge → Arduino):  "M<idx> <pwm>\n"
 *   pwm 0–255 maps to step position [-MAX_STEPS, +MAX_STEPS]
 *   0   = fully negative DOF  (e.g. shoulder fully back)
 *   127 = neutral centre
 *   255 = fully positive DOF  (e.g. shoulder fully forward)
 *
 * Feedback protocol (Arduino → bridge):  "P<idx> <angle>\n"
 *   angle 0–180 represents current position (maps back to DOF [-1, 1])
 *
 * AccelStepper HALF4WIRE:
 *   28BYJ-48 has gear ratio ~64:1 and 8 electrical steps/rev
 *   → 64 × 64 = 4096 half-steps per output shaft revolution
 *   MAX_STEPS = 512 ≈ 45° of output shaft travel each direction (90° total)
 *   Increase MAX_STEPS for more range, decrease for faster response.
 *
 * Requires: AccelStepper library (install via Arduino IDE Library Manager)
 */

#include <AccelStepper.h>

// HALF4WIRE with correct coil order for 28BYJ-48 + ULN2003
// Constructor order: (interface, IN1, IN3, IN2, IN4) — not sequential!
AccelStepper stepper(AccelStepper::HALF4WIRE, 8, 10, 9, 11);

const long           MAX_STEPS    = 512;   // half-steps each direction from centre
const float          MAX_SPEED    = 800.0; // half-steps per second
const float          ACCEL        = 400.0; // half-steps per second²
const unsigned long  WATCHDOG_MS  = 500;   // return to centre if no command
const unsigned long  FEEDBACK_MS  = 50;    // position report interval (~20 Hz)

unsigned long lastCmdMs      = 0;
unsigned long lastFeedbackMs = 0;
String        inputBuffer    = "";

void setup() {
  Serial.begin(115200);
  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCEL);
  stepper.setCurrentPosition(0);  // define current position as centre (DOF 0)
  Serial.println("READY");
}

void loop() {
  unsigned long now = millis();

  // Always call run() — moves one step if needed, non-blocking
  stepper.run();

  // Watchdog: drift back to centre when commands stop arriving
  if (now - lastCmdMs > WATCHDOG_MS) {
    stepper.moveTo(0);
  }

  // Read serial commands
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // Send position feedback at ~20 Hz
  if (now - lastFeedbackMs >= FEEDBACK_MS) {
    long pos   = stepper.currentPosition();
    int  angle = (int)map(pos, -MAX_STEPS, MAX_STEPS, 0, 180);
    angle      = constrain(angle, 0, 180);
    Serial.print("P0 ");
    Serial.println(angle);
    lastFeedbackMs = now;
  }
}

void processCommand(String line) {
  line.trim();
  if (line.length() < 3)       return;
  if (line.charAt(0) != 'M')   return;

  int spaceIdx = line.indexOf(' ');
  if (spaceIdx < 0) return;

  int motorIdx = line.substring(1, spaceIdx).toInt();
  int pwm      = constrain(line.substring(spaceIdx + 1).toInt(), 0, 255);

  if (motorIdx == 0) {
    // Map PWM [0, 255] → step target [-MAX_STEPS, +MAX_STEPS]
    long target = map(pwm, 0, 255, -MAX_STEPS, MAX_STEPS);
    stepper.moveTo(target);
    lastCmdMs = millis();
  }
}
