import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class HardwareDiscoveryDaemon:
    """
    Simulates a USB/EtherCAT daemon that queries smart servos 
    (like ROBOTIS Dynamixel X-series) on a daisy-chain bus.
    
    In a real deployment, this would use PySerial or ROS2 to read 
    the EEPROM of each connected motor, determining its Model, 
    Firmware Version, and Kinematic role, allowing true 
    "Plug-and-Play" topological generation.
    """
    def __init__(self, mock_scan_time: float = 1.0):
        self.mock_scan_time = mock_scan_time
        
    def scan_bus(self) -> List[Dict]:
        """
        Mocks discovering an unmapped 3-DOF extremity 
        (e.g., a newly plugged in prosthetic finger or micro-arm).
        """
        logger.info("[Daemon] Scanning hardware bus (USB/EtherCAT)...")
        time.sleep(self.mock_scan_time)
        
        # Mocks the EEPROM response of 3 daisy-chained servos
        discovered_nodes = [
            {"id": 101, "model": "XL430-W250", "role": "base_pan", "features": [1.0, 0.0, 0.0]},
            {"id": 102, "model": "XL430-W250", "role": "shoulder_tilt", "features": [0.8, 0.2, 0.1]},
            {"id": 103, "model": "XL430-W250", "role": "elbow_tilt", "features": [0.5, 0.5, 0.2]},
        ]
        
        logger.info(f"[Daemon] Discovered {len(discovered_nodes)} new actuator nodes!")
        return discovered_nodes

class BluetoothDiscoveryDaemon:
    """
    Bluetooth LE scanner that detects dynamically connected smart prosthetic extensions.
    Instead of discrete pairing, it listens for our Service UUID and retrieves
    their EEPROM/Kinematic features natively, bridging into the existing KinematicGNN.
    """
    def __init__(self, service_uuid: str = "0000ffe0-0000-1000-8000-00805f9b34fb"):
        self.service_uuid = service_uuid
        
    def scan_bus(self) -> List[Dict]:
        """Scans for BLE devices broadcasting the Noosphere extension service UUID."""
        logger.info("[Daemon] Scanning Bluetooth LE airwaves...")
        import asyncio
        return asyncio.run(self._async_scan())
        
    async def _async_scan(self):
        try:
            from bleak import BleakScanner
        except ImportError:
            logger.error("bleak missing. Cannot scan for Bluetooth nodes.")
            return []
            
        discovered_nodes = []
        try:
            # We scan for 2 seconds
            devices = await BleakScanner.discover(timeout=2.0)
            for d in devices:
                ad = d.metadata.get("uuids", [])
                if any(self.service_uuid.lower() in u.lower() for u in ad):
                    # Mocks discovering a 2-DOF BLE gripper when connected
                    discovered_nodes.extend([
                        {"id": 201, "model": "BLE-GRIPPER", "role": "wrist_roll", "features": [0.2, 0.8, 0.1]},
                        {"id": 202, "model": "BLE-GRIPPER", "role": "gripper_pinch", "features": [0.1, 0.9, 0.0]},
                    ])
                    logger.info(f"[Daemon] Discovered Bluetooth extension node: {d.address}")
                    break  # Found our node
        except Exception as e:
            logger.error(f"[Daemon] BLE scan failed: {e}")
            
        return discovered_nodes
