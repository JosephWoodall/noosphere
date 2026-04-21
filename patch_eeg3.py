import re
with open('noosphere/s4_eeg.py', 'r') as f: content = f.read()

old_loss = """    def forward(self, alpha: torch.Tensor, target_one_hot: torch.Tensor, epoch: int) -> torch.Tensor:
        S           = alpha.sum(-1, keepdim=True)
        pred_probs  = alpha / S
        
        err         = (target_one_hot - pred_probs) ** 2
        var         = (alpha * (S - alpha)) / (S**2 * (S + 1))
        loss_sos    = (err + var).sum(-1)
        
        anneal_coef = min(1.0, epoch / self.annealing_step)
        alpha_reg   = target_one_hot + (1.0 - target_one_hot) * alpha
        kl_reg      = self._kl(alpha_reg)
        
        return (loss_sos + anneal_coef * kl_reg).mean()"""

new_loss = """    def forward(self, alpha: torch.Tensor, target_one_hot: torch.Tensor, epoch: int) -> torch.Tensor:
        S           = alpha.sum(-1, keepdim=True)
        # NLL Digamma formulation for strict, fast probabilistic convergence
        loss_ce     = (target_one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
        
        # Annealed KL Divergence for Epistemic boundaries
        anneal_coef = min(1.0, epoch / self.annealing_step)
        alpha_reg   = target_one_hot + (1.0 - target_one_hot) * alpha
        kl_reg      = self._kl(alpha_reg)
        
        return (loss_ce + anneal_coef * kl_reg).mean()"""

content = content.replace(old_loss, new_loss)

with open('noosphere/s4_eeg.py', 'w') as f: f.write(content)
print("Updated to digamma NLL.")
