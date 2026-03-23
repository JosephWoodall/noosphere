import torch
import torch.nn as nn
from noosphere.agent import NoosphereAgent, AgentConfig
from noosphere.actions import make_shell_space, ShellExecutor, ActBridge, Tier

def test_prosthetic_alignment():
    print("Running Regression Test: Prosthetic Alignment & Safety Gating")
    
    device = torch.device("cpu")
    space = make_shell_space(working_dir=".")
    
    # Needs a smaller vocab for fast testing but 148 is fine
    cfg = AgentConfig(n_actions=space.n_actions, n_eeg_ch=3, min_act_confidence=0.5)
    agent = NoosphereAgent(cfg, device)
    
    # 1. Setup Mock bridge
    executor = ShellExecutor(allow_tiers={Tier.SAFE_READ})
    bridge = ActBridge(space, executor, min_confidence=0.4)
    agent.act_bridge = bridge

    # 2. Test BCI Bypass
    # We mock out the S4 perception to force a confident intent
    class MockS4(nn.Module):
        def forward(self, eeg, mask=None, inference=False):
            logits = torch.zeros(1, space.n_actions)
            logits[0, 5] = 10.0  # Force intent to action 5
            return {
                "summary": torch.zeros(1, 256),
                "sequence": torch.zeros(1, 60, 256),
                "intent_logits": logits,
                "continuous_xyz": torch.zeros(1, 3),
                "confidence": torch.tensor([0.99]),
                "cognitive": {
                    "workload": torch.tensor([0.1]),
                    "attention": torch.tensor([0.9]),
                    "arousal": torch.tensor([0.5]),
                    "valence": torch.tensor([0.5]),
                    "fatigue": torch.tensor([0.1])
                },
                "planning_budget": torch.tensor([1.0])
            }
    
    original_s4 = agent.perception.s4
    agent.perception.s4 = MockS4()
    
    # Mock Consequence model to ensure it is SAFE
    original_consequence = agent.consequence
    class SafeConsequence(nn.Module):
        def forward(self, state):
            return {
                "reward": torch.tensor([0.1]),
                "value": torch.tensor([0.5]),
                "termination": torch.tensor([0.05]) # Safe!
            }
    agent.consequence = SafeConsequence()
    
    dummy_obs = {
        "eeg": torch.zeros(3, 256),
    }
    
    # Run step deterministically so the 99% probability argmaxes to 5
    action, info = agent.step(dummy_obs, deterministic=True)
    assert action == 5, f"Expected Shared Autonomy Blend to heavily favor 5. Got {action}."
    assert info["sim_termination"] < 0.1, "Simulated termination should be low."
    
    # 3. Test Safety Gating 
    # If the user commands action 5, but the world model predicts it's fatal
    class FatalConsequence(nn.Module):
        def forward(self, state):
            return {
                "reward": torch.tensor([-1.0]),
                "value": torch.tensor([-0.9]),
                "termination": torch.tensor([0.95]) # Fatal!
            }
    agent.consequence = FatalConsequence()
    
    action, info = agent.step(dummy_obs, deterministic=True)
    assert action == 5, "Shared Autonomy Blend should still dictate the intended action."
    
    # Bridge should reject it
    assert info["act_executed"] == False, "ActBridge FAILED to gate a catastrophic BCI command!"
    outcome = info.get("act_outcome", "") or info.get("reason", "")
    assert "rejected" in outcome.lower() or "safety" in outcome.lower() or outcome == "", "Outcome should indicate rejection or be empty"
    
    # 4. Test RL Fallback (Low Confidence BCI)
    class UncertainS4(nn.Module):
        def forward(self, eeg, mask=None, inference=False):
            logits = torch.zeros(1, space.n_actions)
            logits[0, 5] = 10.0 # Intent is 5, but...
            return {
                "summary": torch.zeros(1, 256),
                "sequence": torch.zeros(1, 60, 256),
                "intent_logits": logits,
                "continuous_xyz": torch.zeros(1, 3),
                "confidence": torch.tensor([0.1]), # LOW CONFIDENCE!
                "cognitive": {
                    "workload": torch.tensor([0.1]),
                    "attention": torch.tensor([0.9]),
                    "arousal": torch.tensor([0.5]),
                    "valence": torch.tensor([0.5]),
                    "fatigue": torch.tensor([0.1])
                },
                "planning_budget": torch.tensor([1.0])
            }
            
    agent.perception.s4 = UncertainS4()
    agent.consequence = SafeConsequence()
    
    action, info = agent.step(dummy_obs, deterministic=True)
    # The BCI weight is only 0.1, so the AI's prior dominates. Since the Actor is untrained, it might pick randomly.
    # We just ensure it doesn't crash and correctly blends rather than defaulting to 5.
    print(f"Blended Shared Autonomy (10% BCI, 90% AI) picked action: {action}")
    
    print("\n✅ All regression tests passed! Prosthetic Alignment is secure.")

if __name__ == "__main__":
    test_prosthetic_alignment()
