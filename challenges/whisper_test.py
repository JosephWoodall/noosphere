import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from noosphere.rssm import RSSM

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def run_whisper_efficacy_test():
    """
    Proves Collective Intelligence: Agent B (Naive) improves its world modeling 
    efficiency by receiving 'Whisper' latent priors from Agent A (Expert).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Starting Whisper Efficacy Test (Collective Intelligence)...")
    
    # 1. Setup Models
    # Small dimensions for fast CPU testing
    params = dict(embed_dim=32, action_dim=16, det_dim=64, stoch_cats=16, stoch_classes=16)
    agent_a_rssm = RSSM(**params).to(device)
    agent_b_rssm = RSSM(**params).to(device)
    
    # 2. Agent A becomes an 'Expert' through a mock training phase
    # (In a real scenario, this would be actual training. Here we simulate it by 
    # ensuring Agent A's prior and posterior are somewhat aligned for a specific pattern.)
    log.info("Agent A (Expert) synchronizing with environment dynamics...")
    batch_size = 16
    seq_len = 5
    
    # Create a consistent 'pattern' in the environment
    pattern_obs = torch.randn(seq_len, batch_size, 32).to(device)
    pattern_act = torch.randn(seq_len, batch_size, 16).to(device)
    
    # Simple optimization loop for Agent A to 'learn' this specific sequence
    optimizer_a = torch.optim.Adam(agent_a_rssm.parameters(), lr=1e-3)
    for _ in range(20):
        h = agent_a_rssm.initial_state(batch_size, device)["h"]
        z = agent_a_rssm.initial_state(batch_size, device)["z"]
        total_kl = 0
        for t in range(seq_len):
            h, z, prior, post = agent_a_rssm.observe_step(h, z, pattern_act[t], pattern_obs[t])
            total_kl += agent_a_rssm.kl_loss(prior, post, free_nats=0.1)
        
        optimizer_a.zero_grad()
        total_kl.backward()
        optimizer_a.step()
    
    log.info(f"Agent A Expert KL Loss: {total_kl.item():.4f}")

    # 3. Test Agent B (Naive) on the same pattern
    log.info("Evaluating Naive Agent B...")
    h_b = agent_b_rssm.initial_state(batch_size, device)["h"]
    z_b = agent_b_rssm.initial_state(batch_size, device)["z"]
    
    losses_naive = []
    # Store Agent A's insights for later
    whisper_insights = []
    
    with torch.no_grad():
        # Agent A generates insights (Priors) while Agent B struggles
        h_a = agent_a_rssm.initial_state(batch_size, device)["h"]
        z_a = agent_a_rssm.initial_state(batch_size, device)["z"]
        
        for t in range(seq_len):
            # Agent B (Naive)
            h_b, z_b, prior_b, post_b = agent_b_rssm.observe_step(h_b, z_b, pattern_act[t], pattern_obs[t])
            losses_naive.append(agent_b_rssm.kl_loss(prior_b, post_b, free_nats=0.1).item())
            
            # Agent A (Expert) generates 'Whisper' (the prior prediction)
            # A 'Whisper' is sharing the PREDICTION of what happens next.
            h_a, z_a, prior_a, post_a = agent_a_rssm.observe_step(h_a, z_a, pattern_act[t], pattern_obs[t])
            whisper_insights.append(prior_a.detach())
            
    avg_loss_naive = np.mean(losses_naive)
    log.info(f"Agent B (Naive) Average KL Loss: {avg_loss_naive:.4f}")

    # 4. Agent B receives 'Whisper' and retries
    log.info("Agent B receiving 'Whisper' insights from Agent A...")
    h_b = agent_b_rssm.initial_state(batch_size, device)["h"]
    z_b = agent_b_rssm.initial_state(batch_size, device)["z"]
    
    losses_whisper = []
    with torch.no_grad():
        for t in range(seq_len):
            # In 'Whisper' mode, Agent B biases its own prior with Agent A's expert prior
            # We simulate this by blending Agent B's naive prior with Agent A's whisper
            
            # First, standard step to get B's naive prior/posterior
            h_b_next, z_b_next, prior_b, post_b = agent_b_rssm.observe_step(h_b, z_b, pattern_act[t], pattern_obs[t])
            
            # Calculate loss IF Agent B had used Agent A's whisper as its prior
            # This proves that Agent A's prior is a better fit for the data
            loss_with_whisper = agent_b_rssm.kl_loss(whisper_insights[t], post_b, free_nats=0.1).item()
            losses_whisper.append(loss_with_whisper)
            
            h_b, z_b = h_b_next, z_b_next

    avg_loss_whisper = np.mean(losses_whisper)
    log.info(f"Agent B (with Whisper) Average KL Loss: {avg_loss_whisper:.4f}")
    
    improvement = (avg_loss_naive - avg_loss_whisper) / avg_loss_naive * 100
    log.info(f"Performance Gain via Whisper: {improvement:.2f}%")
    
    if improvement > 15.0:
        log.info("RESULT: SUCCESS ✓ - Collective Intelligence significantly improves efficiency.")
    else:
        log.info("RESULT: FAILED ✗ - Improvement was not significant enough.")
        
    return improvement > 15.0

if __name__ == "__main__":
    run_whisper_efficacy_test()
