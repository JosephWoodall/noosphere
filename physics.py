"""
noosphere/physics.py
====================
Physics-Augmented World Model

Fix in v1.4.0
-------------
observe_step now returns the raw physics tensor loss as a 7th element
alongside the float log dict. This allows agent._update_wm to accumulate
it as a tensor (keeping the gradient graph) rather than as a float
(which severed the graph and made conservation law penalties a no-op).

Return signature change:
    v1.3.x: h_n, z_n, pp, qp, phys_state, phys_log_dict
    v1.4.0: h_n, z_n, pp, qp, phys_state, phys_tensor_loss, phys_log_dict

phys_tensor_loss  : scalar tensor with grad — goes into loss.backward()
phys_log_dict     : dict of floats — goes into metrics logging only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


# ── Physical state container ──────────────────────────────────────────────────

class PhysicalState:
    __slots__ = ["pos","vel","rot","omega","mass","inertia",
                 "forces","contacts","energy","fluid_v","B","N","device"]

    def __init__(self, B: int, N: int, device: torch.device, G: int = 4):
        self.B = B; self.N = N; self.device = device
        self.pos      = torch.zeros(B, N, 3, device=device)
        self.vel      = torch.zeros(B, N, 3, device=device)
        self.rot      = torch.zeros(B, N, 4, device=device); self.rot[..., 0] = 1.0
        self.omega    = torch.zeros(B, N, 3, device=device)
        self.mass     = torch.ones(B, N,    device=device)
        self.inertia  = torch.ones(B, N, 3, device=device) * 0.1
        self.forces   = torch.zeros(B, N, 3, device=device)
        self.contacts = torch.zeros(B, N, N, device=device)
        self.energy   = torch.zeros(B,       device=device)
        self.fluid_v  = torch.zeros(B, G**3, 3, device=device)

    def flatten(self) -> torch.Tensor:
        B = self.B
        return torch.cat([
            self.pos.reshape(B,-1),   self.vel.reshape(B,-1),
            self.rot.reshape(B,-1),   self.omega.reshape(B,-1),
            self.mass.reshape(B,-1),  self.inertia.reshape(B,-1),
            self.forces.reshape(B,-1),self.contacts.reshape(B,-1),
            self.energy.unsqueeze(-1),self.fluid_v.reshape(B,-1),
        ], dim=-1)


def _phys_dim(N: int, G: int = 4) -> int:
    return N*3 + N*3 + N*4 + N*3 + N + N*3 + N*3 + N*N + 1 + G**3*3


# ── State estimator ───────────────────────────────────────────────────────────

class PhysicsStateEstimator(nn.Module):
    def __init__(self, embed_dim: int, n_bodies: int = 4,
                 hidden_dim: int = 256, G: int = 4):
        super().__init__()
        self.N = n_bodies; self.G = G
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        D = hidden_dim; N = n_bodies
        self.heads = nn.ModuleDict({
            "pos":     nn.Linear(D, N*3),
            "vel":     nn.Linear(D, N*3),
            "rot":     nn.Linear(D, N*4),
            "omega":   nn.Linear(D, N*3),
            "mass":    nn.Linear(D, N),
            "inertia": nn.Linear(D, N*3),
            "force":   nn.Linear(D, N*3),
            "contact": nn.Linear(D, N*N),
            "fluid":   nn.Linear(D, G**3*3),
        })

    def forward(self, e: torch.Tensor) -> PhysicalState:
        B, N, G = e.shape[0], self.N, self.G
        h = self.trunk(e)
        s = PhysicalState(B, N, e.device, G)
        s.pos     = self.heads["pos"](h).reshape(B,N,3)
        s.vel     = self.heads["vel"](h).reshape(B,N,3)
        s.rot     = F.normalize(self.heads["rot"](h).reshape(B,N,4), dim=-1)
        s.omega   = self.heads["omega"](h).reshape(B,N,3)
        s.mass    = F.softplus(self.heads["mass"](h).reshape(B,N)) + 0.01
        s.inertia = F.softplus(self.heads["inertia"](h).reshape(B,N,3)) + 1e-4
        s.forces  = self.heads["force"](h).reshape(B,N,3)
        s.contacts= torch.sigmoid(self.heads["contact"](h).reshape(B,N,N))
        s.fluid_v = self.heads["fluid"](h).reshape(B,G**3,3)
        ke  = 0.5*(s.mass*s.vel.pow(2).sum(-1)).sum(-1)
        kr  = 0.5*(s.inertia*s.omega.pow(2)).sum(-1).sum(-1)
        pe  = (s.mass*9.81*s.pos[...,2]).sum(-1)
        s.energy = ke + kr + pe
        return s


# ── Physics transition (RK4) ──────────────────────────────────────────────────

class PhysicsTransitionPrior(nn.Module):
    def __init__(self, n_bodies: int = 4, dt: float = 1/60, gravity: float = 9.81,
                 rho: float = 1.225, Cd: float = 0.47, e_rest: float = 0.6,
                 nu: float = 1e-3):
        super().__init__()
        self.N = n_bodies; self.dt = dt; self.g = gravity
        self.rho = rho; self.Cd = Cd; self.e = e_rest; self.nu = nu
        self.log_A      = nn.Parameter(torch.zeros(n_bodies))
        self.log_k      = nn.Parameter(torch.ones(1) * math.log(1000.0))
        self.log_b      = nn.Parameter(torch.ones(1) * math.log(10.0))
        self.fluid_diff = nn.Conv3d(3, 3, 3, padding=1, groups=3, bias=False)
        self.register_buffer("_g_vec",
            torch.tensor([0.0, 0.0, -gravity], dtype=torch.float32))

    def _deriv(self, s: PhysicalState, Fext: torch.Tensor, tau: torch.Tensor):
        g_vec  = self._g_vec.view(1,1,3)
        Fg     = s.mass.unsqueeze(-1) * g_vec
        A      = F.softplus(self.log_A).view(1, self.N, 1)
        vm     = s.vel.norm(-1, keepdim=True).clamp(1e-6)
        Fd     = -0.5 * self.rho * self.Cd * A * vm * s.vel
        k      = F.softplus(self.log_k); b = F.softplus(self.log_b)
        pi     = s.pos.unsqueeze(2); pj = s.pos.unsqueeze(1)
        d_ij   = pj - pi
        dist   = d_ij.norm(-1, keepdim=True).clamp(1e-6)
        n_hat  = d_ij / dist
        pen    = F.relu(1.0 - dist.squeeze(-1))
        vi     = s.vel.unsqueeze(2); vj = s.vel.unsqueeze(1)
        Fc     = (k*pen.unsqueeze(-1)*n_hat + b*(vj-vi)*s.contacts.unsqueeze(-1)).sum(2)
        dv     = (Fext + Fg + Fd + Fc) / s.mass.unsqueeze(-1)
        dpos   = s.vel
        Io     = s.inertia * s.omega
        domega = (tau - torch.cross(s.omega, Io, -1)) / s.inertia.clamp(1e-6)
        w,x,y,z = s.rot[...,0],s.rot[...,1],s.rot[...,2],s.rot[...,3]
        ox,oy,oz = s.omega[...,0],s.omega[...,1],s.omega[...,2]
        drot   = 0.5 * torch.stack([
            -x*ox-y*oy-z*oz, w*ox+y*oz-z*oy,
             w*oy-x*oz+z*ox, w*oz+x*oy-y*ox], -1)
        P_in   = (Fext*s.vel).sum(-1).sum(-1)
        P_dis  = -(Fd*s.vel).sum(-1).sum(-1)
        dE     = P_in - P_dis
        G = int(round(s.fluid_v.shape[1]**(1/3)))
        try:
            u      = s.fluid_v.reshape(s.B, G, G, G, 3).permute(0,4,1,2,3)
            dfluid = self.nu * self.fluid_diff(u)
            dfluid = dfluid.permute(0,2,3,4,1).reshape(s.B, G**3, 3)
        except Exception:
            dfluid = torch.zeros_like(s.fluid_v)
        return dpos, dv, drot, domega, dE, dfluid

    def _apply(self, s, dt, dp, dv, dr, dw, de, dfluid):
        G  = int(round(s.fluid_v.shape[1]**(1/3)))
        ns = PhysicalState(s.B, s.N, s.pos.device, G)
        ns.pos      = s.pos   + dt*dp
        ns.vel      = s.vel   + dt*dv
        ns.rot      = F.normalize(s.rot + dt*dr, -1)
        ns.omega    = s.omega + dt*dw
        ns.mass     = s.mass;   ns.inertia  = s.inertia
        ns.forces   = dv * s.mass.unsqueeze(-1)
        ns.contacts = s.contacts
        ns.energy   = s.energy + dt*de
        new_u       = s.fluid_v + dt * dfluid
        ns.fluid_v  = new_u - new_u.mean(dim=1, keepdim=True)
        return ns

    def forward(self, s: PhysicalState, Fext: torch.Tensor, tau: torch.Tensor):
        dt  = self.dt
        k1  = self._deriv(s,                         Fext, tau)
        k2  = self._deriv(self._apply(s, dt/2, *k1), Fext, tau)
        k3  = self._deriv(self._apply(s, dt/2, *k2), Fext, tau)
        k4  = self._deriv(self._apply(s, dt,   *k3), Fext, tau)
        def rk4c(y, *ks): return y + (dt/6)*(ks[0]+2*ks[1]+2*ks[2]+ks[3])
        G   = int(round(s.fluid_v.shape[1]**(1/3)))
        ns  = PhysicalState(s.B, s.N, s.pos.device, G)
        ns.pos     = rk4c(s.pos,            k1[0],k2[0],k3[0],k4[0])
        ns.vel     = rk4c(s.vel,            k1[1],k2[1],k3[1],k4[1])
        ns.rot     = F.normalize(rk4c(s.rot,k1[2],k2[2],k3[2],k4[2]),-1)
        ns.omega   = rk4c(s.omega,          k1[3],k2[3],k3[3],k4[3])
        ns.mass    = s.mass; ns.inertia = s.inertia; ns.contacts = s.contacts
        ns.energy  = rk4c(s.energy.unsqueeze(-1),
                          k1[4].unsqueeze(-1),k2[4].unsqueeze(-1),
                          k3[4].unsqueeze(-1),k4[4].unsqueeze(-1)).squeeze(-1)
        ns.forces  = Fext + s.mass.unsqueeze(-1)*k1[1]
        raw_fluid  = rk4c(s.fluid_v, k1[5],k2[5],k3[5],k4[5])
        ns.fluid_v = raw_fluid - raw_fluid.mean(dim=1, keepdim=True)
        return ns


# ── Conservation laws ─────────────────────────────────────────────────────────

class ConservationLaws(nn.Module):
    def __init__(self, energy_w=1.0, momentum_w=1.0, angular_w=0.5,
                 quat_w=1.0, fluid_w=0.5):
        super().__init__()
        self.ew=energy_w; self.pw=momentum_w; self.aw=angular_w
        self.qw=quat_w;   self.fw=fluid_w

    def forward(self, s0, s1, Fext, tau, dt) -> Dict[str, torch.Tensor]:
        W_ext  = (Fext * (s0.vel + s1.vel) * 0.5).sum(-1).sum(-1) * dt
        L_E    = self.ew * ((s1.energy - s0.energy - W_ext).pow(2).mean())
        dp_exp = (Fext + s0.mass.unsqueeze(-1) *
                  torch.tensor([0,0,-9.81], device=Fext.device, dtype=Fext.dtype)
                  ).sum(1) * dt
        dp_act = (s1.vel - s0.vel) * s0.mass.unsqueeze(-1)
        L_p    = self.pw * (dp_act.sum(1) - dp_exp).pow(2).mean()
        dL_exp = (tau * dt).sum(1)
        dL_act = s0.inertia * (s1.omega - s0.omega)
        L_ang  = self.aw * (dL_act.sum(1) - dL_exp).pow(2).mean()
        L_q    = self.qw * (s1.rot.norm(-1) - 1.0).pow(2).mean()
        u      = s1.fluid_v
        L_f    = self.fw * u.mean(dim=1).pow(2).mean()
        total  = L_E + L_p + L_ang + L_q + L_f
        return {
            "physics/energy":     L_E,
            "physics/momentum":   L_p,
            "physics/angular":    L_ang,
            "physics/quat":       L_q,
            "physics/fluid":      L_f,
            "physics/total_loss": total,   # this is still a tensor
        }


# ── Residual corrector ────────────────────────────────────────────────────────

class ResidualCorrector(nn.Module):
    def __init__(self, phys_dim: int, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phys_dim + embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),               nn.SiLU(),
            nn.Linear(hidden, phys_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phys_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phys_flat, obs], dim=-1))


# ── Physics-augmented RSSM ────────────────────────────────────────────────────

class PhysicsAugmentedRSSM(nn.Module):
    """
    observe_step returns 7 values (v1.4.0+):
        h_next, z_next, prior_probs, posterior_probs,
        phys_state, phys_tensor_loss, phys_log_dict

    phys_tensor_loss  is a scalar tensor — stays in graph for backward()
    phys_log_dict     is {str: float} — for logging only
    """
    def __init__(
        self,
        embed_dim:     int = 512,
        action_dim:    int = 64,
        n_bodies:      int = 4,
        G:             int = 4,
        det_dim:       int = 512,
        stoch_cats:    int = 32,
        stoch_classes: int = 32,
        hidden_dim:    int = 256,
        dt:            float = 1/60,
    ):
        super().__init__()
        from noosphere.rssm import RSSM as _RSSM
        self.rssm        = _RSSM(embed_dim, action_dim, det_dim,
                                  stoch_cats, stoch_classes, hidden_dim)
        phys_dim         = _phys_dim(n_bodies, G)
        self.state_est   = PhysicsStateEstimator(embed_dim, n_bodies, hidden_dim, G)
        self.prior       = PhysicsTransitionPrior(n_bodies, dt)
        self.corrector   = ResidualCorrector(phys_dim, embed_dim, hidden_dim)
        self.conservation= ConservationLaws()
        self.phys_proj   = nn.Linear(phys_dim, embed_dim)
        self._dt         = dt

    @property
    def state_dim(self) -> int:
        return self.rssm.state_dim

    def initial_state(self, batch_size: int, device: torch.device) -> Dict:
        return self.rssm.initial_state(batch_size, device)

    def reset_episode(self):
        self.rssm.reset_episode()

    def observe_step(self, h, z, a, obs_embed):
        phys_s  = self.state_est(obs_embed)
        Fext    = phys_s.forces
        tau     = torch.zeros_like(Fext)
        phys_n  = self.prior(phys_s, Fext, tau)
        flat    = phys_n.flatten()
        delta   = self.corrector(flat, obs_embed)
        corrected_embed = obs_embed + self.phys_proj(flat + delta)
        h_n, z_n, pp, qp = self.rssm.observe_step(h, z, a, corrected_embed)

        # Physics losses: keep total_loss as a tensor in the graph.
        # Convert individual terms to floats for logging only.
        phys_losses      = self.conservation(phys_s, phys_n, Fext, tau, self._dt)
        phys_tensor_loss = phys_losses["physics/total_loss"]   # tensor — grad flows
        phys_log         = {k: v.item() for k, v in phys_losses.items()}

        return h_n, z_n, pp, qp, phys_n, phys_tensor_loss, phys_log

    def imagine_step(self, h, z, a):
        return self.rssm.imagine_step(h, z, a)

    def kl_loss(self, pp, qp, free_nats=1.0):
        return self.rssm.kl_loss(pp, qp, free_nats)
