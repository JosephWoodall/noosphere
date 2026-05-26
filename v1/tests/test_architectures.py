import torch
import torch.nn as nn
import pytest

import os
import sys

sys.path.insert(0, '/home/redleadr/workspace')

from noosphere.s4_eeg import S4EEGEncoder
from noosphere.gnn import KinematicGNN
from noosphere.physics import PhysicsAugmentedRSSM
from noosphere.rssm import ConsequenceModel
from noosphere.planner import Actor, MCTSPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_s4_eeg_encoder_edl():
    """Verify S4EEGEncoder output shapes and Evidential Dirichlet parameters."""
    B, C, T = 2, 8, 256
    n_intent = 5
    encoder = S4EEGEncoder(n_channels=C, d_model=32, n_intent=n_intent).to(device)
    
    # Mock EEG sequence
    eeg = torch.randn(B, C, T, device=device)
    mask = torch.ones(B, C, device=device)
    
    out = encoder(eeg)
    
    # Check EDL math property: probabilities sum to 1
    p_sum = out["intent_probs"].sum(dim=-1)
    assert torch.allclose(p_sum, torch.ones_like(p_sum)), "EDL probabilities must sum to 1.0"
    
    # Check uncertainty limits
    assert torch.all(out["uncertainty"] >= 0.0) and torch.all(out["uncertainty"] <= 1.0), "EDL uncertainty must be [0, 1]"
    assert torch.all(out["confidence"] >= 0.0) and torch.all(out["confidence"] <= 1.0), "EDL confidence must be [0, 1]"
    
    # Check shapes
    assert out["intent_probs"].shape == (B, n_intent)
    assert out["evidence"].shape == (B, n_intent)
    assert out["alpha"].shape == (B, n_intent)
    
    # Check alpha boundaries (alpha = e + 1, e >= 0 => alpha >= 1)
    assert torch.all(out["alpha"] >= 1.0), "Dirichlet alpha parameters must be >= 1.0"

def test_kinematic_gnn():
    B, N_NODES, D_IN = 2, 4, 16
    gnn = KinematicGNN(n_nodes=N_NODES, node_feature_dim=D_IN, d_model=32).to(device)
    
    # Provide node features
    nodes = torch.randn(B, N_NODES, D_IN, device=device)
    out_dict = gnn(nodes)
    
    assert out_dict["graph_sequence"].shape == (B, N_NODES, 32)
    assert out_dict["graph_token"].shape == (B, 1, 32)

def test_physics_rssm():
    B = 4
    D_MOD, D_DET, D_STOCH, D_ACT = 32, 64, 16, 8
    
    rssm = PhysicsAugmentedRSSM(
        embed_dim=D_MOD,
        det_dim=D_DET,
        stoch_cats=D_STOCH,
        stoch_classes=4,
        action_dim=D_ACT,
        n_bodies=3
    ).to(device)
    
    obs_embed = torch.randn(B, D_MOD, device=device)
    action = torch.randn(B, D_ACT, device=device)
    init_s = rssm.rssm.initial_state(B, device)
    prev_h, prev_z = init_s["h"], init_s["z"]
    
    h, z, *_ = rssm.rssm.observe_step(prev_h, prev_z, action, obs_embed)
    
    assert h.shape == (B, D_DET)
    assert z.shape == (B, D_STOCH * 4)

def test_consequence_model():
    B, D_DET, D_STOCH = 4, 64, 16
    latent_dim = D_DET + (D_STOCH * 4)
    
    critic = ConsequenceModel(state_dim=latent_dim, hidden_dim=128).to(device)
    
    state = torch.randn(B, latent_dim, device=device)
    out = critic(state)
    
    assert "reward" in out
    assert "value" in out
    assert "termination" in out
    
    assert out["reward"].shape == (B, 1)
    assert torch.all(out["termination"] >= 0.0) and torch.all(out["termination"] <= 1.0)

def test_actor_digital_twin():
    B, D_DET, D_STOCH = 2, 64, 16
    latent_dim = D_DET + (D_STOCH * 4)
    N_ACTIONS = 5
    
    actor = Actor(state_dim=latent_dim, n_actions=N_ACTIONS).to(device)
    state = torch.randn(B, latent_dim, device=device)
    
    dist, cont = actor(state)
    assert dist.probs.shape == (B, N_ACTIONS)
    assert dist.logits.shape == (B, N_ACTIONS)

"""
We will omit the MCTSPlanner full integration test from unit testing because it 
requires mocking the entire NoosphereAgent, action_encoder, and World Model wrappers 
simultaneously, which is already handled inside the existing regression test suite 
(`tests/test_prosthetic_alignment.py`) where MCTS Shared Autonomy logic is proven.
"""

def test_hardware_discovery_gnn_injection():
    # Simulate a core BCI graph with 4 initial nodes
    device = torch.device("cpu")
    INITIAL_NODES = 4
    D_IN = 16
    
    gnn = KinematicGNN(n_nodes=INITIAL_NODES, node_feature_dim=D_IN, d_model=32).to(device)
    
    # Simulate the HardwareDiscoveryDaemon finding 3 new USB extremities
    NEW_NODES = 3
    gnn.inject_nodes(NEW_NODES)
    
    # Assert the internal node count expanded
    assert gnn.n_nodes == INITIAL_NODES + NEW_NODES
    
    # Verify the Adjacency Matrices expanded to 7x7
    for adj in gnn.adjacencies:
        assert adj.W.shape == (7, 7)
    
    # Formulate a forward pass with the NEW graph size
    B = 2
    nodes = torch.randn(B, 7, D_IN, device=device)
    out_dict = gnn(nodes)
    
    # Assert sequence output matches the new Node count
    assert out_dict["graph_sequence"].shape == (B, 7, 32)
    assert out_dict["graph_token"].shape == (B, 1, 32)
    print("✓ Hardware Discovery Topological Injection verified.")

if __name__ == "__main__":
    print("Running Test Suite: Noosphere Core Neural Models...")
    test_s4_eeg_encoder_edl()
    print("✓ S4EEGEncoder (EDL) bounds verified.")
    test_kinematic_gnn()
    print("✓ KinematicGNN graphs verified.")
    test_physics_rssm()
    print("✓ PhysicsAugmentedRSSM dimensions verified.")
    test_consequence_model()
    print("✓ ConsequenceModel bounds verified.")
    test_actor_digital_twin()
    test_hardware_discovery_gnn_injection()
    print("✓ Actor Policy outputs verified.")
    print("\nALL NEURAL MODELS PASSED.")
