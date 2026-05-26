/*
 * motor_control.ino — Digital Twin Arduino bridge
 *
 * Receives serial commands from arduino_bridge.py and drives PWM motor outputs.
 * Protocol: "M<index> <pwm>\n"  where pwm is 0–255.
 *
 * Pin map (change to match your wiring):
 *   M0 (shoulder_yaw)   → pin 9   (PWM)
 *   M1 (grip_aperture)  → pin 10  (PWM)
 *
 * Watchdog: if no command received in WATCHDOG_MS, all motors stop.
 */

#define MOTOR_COUNT 2
const int MOTOR_PINS[MOTOR_COUNT] = {9, 10};
const unsigned long WATCHDOG_MS = 500;

unsigned long lastCmdMs = 0;
String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < MOTOR_COUNT; i++) {
    pinMode(MOTOR_PINS[i], OUTPUT);
    analogWrite(MOTOR_PINS[i], 0);
  }
  Serial.println("READY");
}

void loop() {
  // Watchdog: stop all motors if no command received recently
  if (millis() - lastCmdMs > WATCHDOG_MS) {
    for (int i = 0; i < MOTOR_COUNT; i++) {
      analogWrite(MOTOR_PINS[i], 0);
    }
  }

  // Read serial — accumulate until newline
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String line) {
  line.trim();
  if (line.length() < 3) return;
  if (line.charAt(0) != 'M') return;

  int spaceIdx = line.indexOf(' ');
  if (spaceIdx < 0) return;

  int motorIdx = line.substring(1, spaceIdx).toInt();
  int pwm      = line.substring(spaceIdx + 1).toInt();
  pwm = constrain(pwm, 0, 255);

  if (motorIdx >= 0 && motorIdx < MOTOR_COUNT) {
    analogWrite(MOTOR_PINS[motorIdx], pwm);
    lastCmdMs = millis();
  }
}
