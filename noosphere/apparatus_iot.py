import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch

class IoTApparatus:
    """
    Bridge between Noosphere and Smart Home / IoT systems (e.g., Home Assistant).
    Translates digital intent into API calls and observes state changes.
    """
    def __init__(self, api_endpoint: Optional[str] = None, token: Optional[str] = None):
        self.api_endpoint = api_endpoint
        self.token = token
        self.log = logging.getLogger(__name__)
        
        # Internal state cache: device_id -> state_dict
        self.state_cache: Dict[str, Any] = {
            "light.living_room": {"state": "off", "brightness": 0},
            "lock.front_door": {"state": "locked"},
            "switch.coffee_maker": {"state": "off"}
        }

    def get_state_vector(self) -> np.ndarray:
        """Encodes current IoT state into a normalized vector for the World Model."""
        # Simple encoding: [light_on, lock_secured, coffee_on]
        vector = []
        vector.append(1.0 if self.state_cache["light.living_room"]["state"] == "on" else 0.0)
        vector.append(1.0 if self.state_cache["lock.front_door"]["state"] == "locked" else 0.0)
        vector.append(1.0 if self.state_cache["switch.coffee_maker"]["state"] == "on" else 0.0)
        return np.array(vector, dtype=np.float32)

    def execute(self, action_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a digital action (e.g., TOGGLE_LIGHT)."""
        self.log.info(f"[IoT] Executing {action_id} with {payload}")
        
        # In mock mode, we just update the cache
        target = payload.get("entity_id")
        if target in self.state_cache:
            if action_id == "TOGGLE":
                current = self.state_cache[target]["state"]
                new_state = "on" if current == "off" else "off"
                self.state_cache[target]["state"] = new_state
            elif action_id == "UNLOCK":
                self.state_cache[target]["state"] = "unlocked"
            elif action_id == "LOCK":
                self.state_cache[target]["state"] = "locked"
                
            return {"status": "success", "new_state": self.state_cache[target]}
            
        return {"status": "error", "message": "Device not found"}

class IoTExecutor:
    """Wraps IoTApparatus for the ActBridge."""
    def __init__(self, apparatus: IoTApparatus):
        self.apparatus = apparatus

    def act(self, action_index: int, **kwargs) -> Dict[str, Any]:
        # Mapping action_index to specific IoT commands
        # For demo, let's assume index 0 = Toggle Light, 1 = Unlock Door
        if action_index == 0:
            return self.apparatus.execute("TOGGLE", {"entity_id": "light.living_room"})
        elif action_index == 1:
            return self.apparatus.execute("UNLOCK", {"entity_id": "lock.front_door"})
        return {"status": "ignored"}
