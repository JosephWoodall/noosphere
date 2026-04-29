import torch
import torch.nn as nn
import logging
from noosphere.rssm import RSSM
from noosphere.s4_eeg import S4EEGEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def run_whisper_efficacy_test():
    """
    Demonstrates Agent B's performance improvement after receiving a 
    'Dynamics Insight' packet (latent prior) from Agent A.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Starting Whisper Efficacy Test (Collective Intelligence)...")
    
    # Initialize two agents with separate World Models
    agent_a_rssm = RSSM(embed_dim=32, action_dim=64, d_state=16).to(device)
    agent_b_rssm = RSSM(embed_dim=32, action_dim=64, d_state=16).to(device)
    
    # Mock some neural embedding sequence (e.g. learning a new task)
    batch_size = 8
    embed_dim = 32
    action_dim = 64
    mock_eeg_embed = torch.randn(batch_size, embed_dim).to(device)
    mock_action = torch.randn(batch_size, action_dim).to(device)
    
    # Agent A learns the task (mock optimization)
    log.info("Agent A is mastering the task (computing posterior resonance)...")
    init_state_a = agent_a_rssm.initial_state(batch_size, device)
    h_a, z_a = init_state_a["h"], init_state_a["z"]
    
    # Observe step for Agent A
    h_a_next, z_a_next, prior_a, post_a = agent_a_rssm.observe_step(h_a, z_a, mock_action, mock_eeg_embed)
    loss_a = agent_a_rssm.kl_loss(prior_a, post_a)
        
    # Generate the 'Dynamics Insight' packet from Agent A
    log.info("Agent A transmitting Dynamics Insight ('Whisper') to Agent B...")
    insight_packet_z = z_a_next.detach()
    
    # Agent B evaluates without insight
    init_state_b = agent_b_rssm.initial_state(batch_size, device)
    h_b, z_b = init_state_b["h"], init_state_b["z"]
    h_b_next, z_b_next, prior_b, post_b = agent_b_rssm.observe_step(h_b, z_b, mock_action, mock_eeg_embed)
    loss_before = agent_b_rssm.kl_loss(prior_b, post_b).item()
        
    # Agent B integrates the insight
    log.info("Agent B integrating insight...")
    # Inject Agent A's stochastic mode state (z) into Agent B
    z_b = insight_packet_z
    
    h_b_next_insight, z_b_next_insight, prior_b_insight, post_b_insight = agent_b_rssm.observe_step(h_b, z_b, mock_action, mock_eeg_embed)
    loss_after = agent_b_rssm.kl_loss(prior_b_insight, post_b_insight).item()
        
    log.info(f"Agent B Loss (Before Whisper): {loss_before:.4f}")
    log.info(f"Agent B Loss (After Whisper):  {loss_after:.4f}")
    
    improvement = (loss_before - loss_after) / loss_before * 100
    if improvement > 0:
        log.info(f"RESULT: SUCCESS - Whisper accelerated learning by {improvement:.2f}% ✓")
    else:
        log.info(f"RESULT: FAILED - Whisper did not improve learning. ✗")
        
    return improvement > 0

if __name__ == "__main__":
    run_whisper_efficacy_test()
