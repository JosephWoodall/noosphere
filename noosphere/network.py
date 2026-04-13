import logging
import time
from enum import IntEnum
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch

class NetworkMessageMacro(IntEnum):
    STATUS_OK = 0
    NEED_HELP = 1
    BUSY = 2
    OPEN_DOOR = 3
    CLOSE_DOOR = 4
    SYNC_REQUEST = 5

    @classmethod
    def get_description(cls, macro: int) -> str:
        descriptions = {
            cls.STATUS_OK: "I'm OK / Acknowledged",
            cls.NEED_HELP: "I need assistance",
            cls.BUSY: "Currently busy / Do not disturb",
            cls.OPEN_DOOR: "Request: Open door",
            cls.CLOSE_DOOR: "Request: Close door",
            cls.SYNC_REQUEST: "Request: Dynamics Insight Sync"
        }
        return descriptions.get(macro, "Unknown Message")

class NetworkSession:
    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.start_time = time.time()
        self.last_activity = time.time()
        self.message_history: List[Dict] = []

    def log_message(self, direction: str, macro: int):
        self.message_history.append({
            "ts": time.time(),
            "direction": direction, # "sent" or "received"
            "macro": macro,
            "text": NetworkMessageMacro.get_description(macro)
        })
        self.last_activity = time.time()

class NetworkSessionManager:
    """Manages 'Brain-Phone' sessions and the Whisper protocol."""
    def __init__(self, local_node_id: str, transport: Any):
        self.local_node_id = local_node_id
        self.transport = transport
        self.active_session: Optional[NetworkSession] = None
        self.session_timeout = 30.0 # Auto-close after 30s of silence
        self.log = logging.getLogger(__name__)
        
        # Identity Confidence Thresholds
        self.open_threshold = 0.85
        self.keep_alive_threshold = 0.40

    def update(self, contact_id: Optional[str], identity_conf: float):
        """Processes neural identity hits to manage session state."""
        now = time.time()
        
        # 1. Check for Session Opening
        if self.active_session is None:
            if contact_id and identity_conf >= self.open_threshold:
                self.active_session = NetworkSession(contact_id)
                self.log.info(f"[Network] Neural Session OPENED with {contact_id}")
        
        # 2. Check for Session Maintenance / Switching
        elif self.active_session:
            # If thinking about someone else strongly, switch? 
            # For now, let's stick to the current peer until timeout or manual close
            if contact_id == self.active_session.peer_id:
                if identity_conf >= self.keep_alive_threshold:
                    self.active_session.last_activity = now
            
            # 3. Handle Timeout
            if now - self.active_session.last_activity > self.session_timeout:
                self.log.info(f"[Network] Neural Session TIMED OUT with {self.active_session.peer_id}")
                self.active_session = None

    def send_message(self, macro: int):
        if not self.active_session:
            self.log.warning("[Network] Attempted to send message without active session")
            return
            
        from noosphere.proto import NCPEncoder
        encoder = NCPEncoder()
        
        # In v1.7.0 we use JSON payloads for IDENTITY/IOT/CONTEXT types within NCP frames
        # We can reuse IDENTITY packet type for simple messaging or define a new one
        # Let's use the MsgType.IDENTITY we added earlier for the session handshaking
        # and we can add MsgType.MESSAGE for actual content
        
        # Actually, let's keep it simple: send an IOT_ACTION frame with a "MESSAGE" type
        # Or just use the transport's peer routing
        
        payload = {"macro": int(macro), "text": NetworkMessageMacro.get_description(macro)}
        frame = encoder.iot_action_packet(
            entity_id=f"peer:{self.active_session.peer_id}", 
            action="MESSAGE", 
            payload=payload
        )
        
        self.transport.publish("ncp:network", frame, peer_id=self.active_session.peer_id)
        self.active_session.log_message("sent", macro)
        self.log.info(f"[Network] Sent to {self.active_session.peer_id}: {payload['text']}")

    def share_insights(self, weights: Dict[str, torch.Tensor]):
        """The 'Whisper' Protocol: Share Dynamics Insights."""
        # Note: Centralized/Decentralized logic happens here
        # We package abstract dynamics (residual corrector)
        # and publish to the global channel or specific peers
        from noosphere.proto import NCPEncoder
        encoder = NCPEncoder()
        
        # Convert tensors to list/numpy for JSON serialization (Insight prototype)
        # In production, this would be a more efficient binary blob
        summary = {k: v.mean().item() for k, v in weights.items() if "weight" in k}
        
        frame = encoder.context_insight_packet("dynamics_residual", {"summary": summary})
        self.transport.publish("ncp:insights", frame)
        self.log.info("[Network] Shared Dynamics Insight 'Whisper' to network")

class NetworkUI:
    """Terminal-based side window for Noosphere messaging."""
    def __init__(self, manager: NetworkSessionManager):
        self.manager = manager
        self.last_draw = 0

    def render(self):
        """Simple text-based render. In a real CLI this would write to a separate buffer."""
        # Only render if state changed significantly or every 2s
        now = time.time()
        if now - self.last_draw < 1.0: return
        self.last_draw = now
        
        print("\n" + "="*40)
        print(" NOOSPHERE NEURAL MESSAGING HUB ")
        print("="*40)
        
        session = self.manager.active_session
        if session:
            print(f" STATUS: CONNECTED TO [{session.peer_id}]")
            print(f" UPTIME: {int(now - session.start_time)}s")
            print("-" * 40)
            for msg in session.message_history[-5:]: # show last 5
                dir_str = ">>" if msg["direction"] == "sent" else "<<"
                print(f" {dir_str} {msg['text']}")
        else:
            print(" STATUS: IDLE (Focus on contact to connect)")
        
        print("="*40 + "\n")
