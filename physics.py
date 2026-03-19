"""
noosphere/physics.py
====================
Physics-Augmented World Model

Hard-codes known physical laws as differentiable constraints layered on
top of the RSSM. Architecture:

    Observation embedding
        ↓
    PhysicsStateEstimator   →  explicit (pos, vel, rot, ω, F, contacts, fluid)
        ↓
    PhysicsTransitionPrior  →  RK4 ODE integration (Newton + drag + contact + NS)
        ↓
    ResidualCorrector       →  Δs = actual - physics  (learns the gap)
        ↓
    ConservationLaws        →  penalty if ΔE, Δp, ΔL, ∇·u violated
        ↓
    PhysicsAugmentedRSSM    →  wraps standard RSSM with physics pathway

The residual-first design means the neural network only needs to learn
what physics misses (contact discontinuities, turbulence, soft-body memory)
rather than rediscovering Newton's laws from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


# ── Physical state container ──────────────────────────────────────────────────

class PhysicalState:
    """Structured physical state for N bodies."""

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
    """Maps observation embedding → explicit PhysicalState."""
    def __init__(self, embed_dim: int, n_bodies: int = 4, G: int = 4, hidden: int = 512):
        super().__init__()
        N, self.N, self.G = n_bodies, n_bodies, G
        self.trunk = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.heads = nn.ModuleDict({
            "pos":    nn.Linear(hidden, N*3),
            "vel":    nn.Linear(hidden, N*3),
            "rot":    nn.Linear(hidden, N*4),
            "omega":  nn.Linear(hidden, N*3),
            "mass":   nn.Linear(hidden, N),
            "inertia":nn.Linear(hidden, N*3),
            "force":  nn.Linear(hidden, N*3),
            "contact":nn.Linear(hidden, N*N),
            "fluid":  nn.Linear(hidden, G**3*3),
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
    """
    Hard-coded Newtonian dynamics, integrated with RK4.

    Equations
    ---------
    Linear:    v̇ = (F_ext + F_grav + F_drag + F_contact) / m
    Rotational:ω̇ = I⁻¹(τ - ω × Iω)
    Quaternion:q̇ = ½ q ⊗ [0, ω]
    Fluid:     ∂u/∂t ≈ ν∇²u  (coarse-grid, learned diffusion kernel)
    """
    def __init__(self, n_bodies: int = 4, dt: float = 1/60, gravity: float = 9.81,
                 rho: float = 1.225, Cd: float = 0.47, e_rest: float = 0.6,
                 nu: float = 1e-3):
        super().__init__()
        self.N = n_bodies; self.dt = dt; self.g = gravity
        self.rho = rho; self.Cd = Cd; self.e = e_rest; self.nu = nu
        self.log_A = nn.Parameter(torch.zeros(n_bodies))
        self.log_k = nn.Parameter(torch.ones(1) * math.log(1000.0))
        self.log_b = nn.Parameter(torch.ones(1) * math.log(10.0))
        self.fluid_diff = nn.Conv3d(3, 3, 3, padding=1, groups=3, bias=False)

    def _deriv(self, s: PhysicalState, Fext: torch.Tensor, tau: torch.Tensor):
        g_vec = torch.tensor([0,0,-self.g], device=s.pos.device, dtype=s.pos.dtype)
        Fg    = s.mass.unsqueeze(-1) * g_vec.view(1,1,3)
        A     = F.softplus(self.log_A).view(1, self.N, 1)
        vm    = s.vel.norm(-1, keepdim=True).clamp(1e-6)
        Fd    = -0.5 * self.rho * self.Cd * A * vm * s.vel
        k     = F.softplus(self.log_k); b = F.softplus(self.log_b)
        pi    = s.pos.unsqueeze(2); pj = s.pos.unsqueeze(1)
        d_ij  = pj - pi
        dist  = d_ij.norm(-1, keepdim=True).clamp(1e-6)
        n_hat = d_ij / dist
        pen   = F.relu(1.0 - dist.squeeze(-1))
        vi    = s.vel.unsqueeze(2); vj = s.vel.unsqueeze(1)
        Fc    = (k*pen.unsqueeze(-1)*n_hat + b*(vj-vi)*s.contacts.unsqueeze(-1)).sum(2)
        dv    = (Fext + Fg + Fd + Fc) / s.mass.unsqueeze(-1)
        dpos  = s.vel
        Io    = s.inertia * s.omega
        domega= (tau - torch.cross(s.omega, Io, -1)) / s.inertia.clamp(1e-6)
        w,x,y,z = s.rot[...,0],s.rot[...,1],s.rot[...,2],s.rot[...,3]
        ox,oy,oz = s.omega[...,0],s.omega[...,1],s.omega[...,2]
        drot  = 0.5 * torch.stack([
            -x*ox-y*oy-z*oz,  w*ox+y*oz-z*oy,
             w*oy-x*oz+z*ox,  w*oz+x*oy-y*ox], -1)
        P_in  = (Fext*s.vel).sum(-1).sum(-1)
        P_dis = -(Fd*s.vel).sum(-1).sum(-1)
        dE    = P_in - P_dis
        return dpos, dv, drot, domega, dE

    def _apply(self, s, dt, dp, dv, dr, dw, de):
        G = int(round(s.fluid_v.shape[1]**(1/3)))
        ns = PhysicalState(s.B, s.N, s.pos.device, G)
        ns.pos     = s.pos  + dt*dp
        ns.vel     = s.vel  + dt*dv
        ns.rot     = F.normalize(s.rot + dt*dr, -1)
        ns.omega   = s.omega+ dt*dw
        ns.mass    = s.mass;  ns.inertia  = s.inertia
        ns.forces  = dv * s.mass.unsqueeze(-1)
        ns.contacts= s.contacts
        ns.energy  = s.energy + dt*de
        try:
            u  = s.fluid_v.reshape(s.B, G, G, G, 3).permute(0,4,1,2,3)
            u2 = u + dt*self.nu*self.fluid_diff(u)
            u2 = u2 - u2.mean([2,3,4], keepdim=True)
            ns.fluid_v = u2.permute(0,2,3,4,1).reshape(s.B, G**3, 3)
        except Exception:
            ns.fluid_v = s.fluid_v
        return ns

    def forward(self, s: PhysicalState, Fext: torch.Tensor, tau: torch.Tensor):
        dt = self.dt
        k1 = self._deriv(s, Fext, tau)
        k2 = self._deriv(self._apply(s, dt/2, *k1), Fext, tau)
        k3 = self._deriv(self._apply(s, dt/2, *k2), Fext, tau)
        k4 = self._deriv(self._apply(s, dt,   *k3), Fext, tau)
        def rk4(y, *ks): return y + (dt/6)*(ks[0]+2*ks[1]+2*ks[2]+ks[3])
        ns = PhysicalState(s.B, s.N, s.pos.device, int(round(s.fluid_v.shape[1]**(1/3))))
        ns.pos    = rk4(s.pos,   k1[0],k2[0],k3[0],k4[0])
        ns.vel    = rk4(s.vel,   k1[1],k2[1],k3[1],k4[1])
        ns.rot    = F.normalize(rk4(s.rot,k1[2],k2[2],k3[2],k4[2]),-1)
        ns.omega  = rk4(s.omega, k1[3],k2[3],k3[3],k4[3])
        ns.mass   = s.mass;  ns.inertia  = s.inertia;  ns.contacts = s.contacts
        ns.energy = rk4(s.energy.unsqueeze(-1),
                        k1[4].unsqueeze(-1),k2[4].unsqueeze(-1),
                        k3[4].unsqueeze(-1),k4[4].unsqueeze(-1)).squeeze(-1)
        ns.forces = Fext + s.mass.unsqueeze(-1)*k1[1]
        ns.fluid_v= self._apply(s, dt, *k1).fluid_v
        return ns


# ── Residual corrector ────────────────────────────────────────────────────────

class ResidualCorrector(nn.Module):
    """
    Learns Δs = s_actual - s_physics.
    Captures contact discontinuities, turbulence, soft-body effects.
    Initialized near-zero; scale grows only if physics is wrong.
    """
    def __init__(self, state_dim: int, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.net   = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phys_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([phys_flat, obs], -1))
        return torch.sigmoid(self.scale) * 0.5 * raw

    def regularisation(self, corr: torch.Tensor) -> torch.Tensor:
        return 0.01 * corr.pow(2).mean()


# ── Conservation law losses ───────────────────────────────────────────────────

class ConservationLaws(nn.Module):
    """
    Differentiable physics constraint losses. These are mathematical
    identities — not data-driven. Enforcing them during training ensures
    physical consistency on out-of-distribution observations.
    """
    def __init__(self, dt: float = 1/60):
        super().__init__()
        self.dt = dt

    def energy(self, s0: PhysicalState, s1: PhysicalState, Fext: torch.Tensor) -> torch.Tensor:
        W = (Fext * s0.vel).sum(-1).sum(-1) * self.dt
        return (s1.energy - s0.energy - W).pow(2).mean()

    def momentum(self, s0: PhysicalState, s1: PhysicalState, Fext: torch.Tensor) -> torch.Tensor:
        p0 = (s0.mass.unsqueeze(-1)*s0.vel).sum(1)
        p1 = (s1.mass.unsqueeze(-1)*s1.vel).sum(1)
        return (p1 - p0 - (Fext*self.dt).sum(1)).pow(2).sum(-1).mean()

    def angular(self, s0: PhysicalState, s1: PhysicalState, tau: torch.Tensor) -> torch.Tensor:
        def L(s): return (s.inertia*s.omega).sum(1) + torch.cross(s.pos, s.mass.unsqueeze(-1)*s.vel,-1).sum(1)
        return (L(s1) - L(s0) - (tau*self.dt).sum(1)).pow(2).sum(-1).mean()

    def quaternion(self, s: PhysicalState) -> torch.Tensor:
        return (s.rot.norm(dim=-1) - 1.0).pow(2).mean()

    def incompressibility(self, s: PhysicalState) -> torch.Tensor:
        G = int(round(s.fluid_v.shape[1]**(1/3)))
        if G < 2: return torch.tensor(0.0, device=s.pos.device)
        u = s.fluid_v.reshape(s.B, G, G, G, 3)
        div = (u[:,1:,:,:,0]-u[:,:-1,:,:,0] +
               u[:,:,1:,:,1]-u[:,:,:-1,:,1] +
               u[:,:,:,1:,2]-u[:,:,:,:-1,2])
        return div.pow(2).mean()

    def total(self, s0, s1, Fext, tau, w=None) -> Tuple[torch.Tensor, Dict[str,float]]:
        w = w or dict(energy=0.1, momentum=0.1, angular=0.05, quaternion=1.0, fluid=0.05)
        losses = {
            "energy":     self.energy(s0, s1, Fext),
            "momentum":   self.momentum(s0, s1, Fext),
            "angular":    self.angular(s0, s1, tau),
            "quaternion": self.quaternion(s1),
            "fluid":      self.incompressibility(s1),
        }
        total = sum(w[k]*v for k,v in losses.items())
        return total, {f"physics/{k}": v.item() for k,v in losses.items()}


# ── Physics-augmented RSSM ────────────────────────────────────────────────────

class PhysicsAugmentedRSSM(nn.Module):
    """
    Wraps a standard RSSM with physics-informed state estimation and
    conservation-law constraints.

    The observe_step additionally returns:
        phys_state   — estimated PhysicalState at this timestep
        phys_losses  — dict of conservation violation scalars
    """
    def __init__(self, embed_dim: int, action_dim: int, n_bodies: int = 4,
                 G: int = 4, det_dim: int = 512, stoch_cats: int = 32,
                 stoch_classes: int = 32, hidden_dim: int = 256, dt: float = 1/60):
        super().__init__()
        from noosphere.rssm import RSSM
        self.N = n_bodies

        self.estimator  = PhysicsStateEstimator(embed_dim, n_bodies, G, hidden_dim)
        self.prior_ode  = PhysicsTransitionPrior(n_bodies, dt)
        self.laws       = ConservationLaws(dt)

        phys_flat = _phys_dim(n_bodies, G)
        self.corrector  = ResidualCorrector(phys_flat, embed_dim, hidden_dim)
        self.F_head     = nn.Linear(action_dim, n_bodies * 3)
        self.tau_head   = nn.Linear(action_dim, n_bodies * 3)
        self.phys_proj  = nn.Sequential(
            nn.Linear(phys_flat, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.rssm = RSSM(embed_dim, action_dim, det_dim, stoch_cats, stoch_classes, hidden_dim)
        self._prev: Optional[PhysicalState] = None

    def initial_state(self, B, dev): return self.rssm.initial_state(B, dev)
    def imagine_step(self, h, z, a): return self.rssm.imagine_step(h, z, a)
    def kl_loss(self, *a, **kw):     return self.rssm.kl_loss(*a, **kw)
    @property
    def state_dim(self):             return self.rssm.state_dim

    def observe_step(self, h, z, a, obs_embed):
        B = obs_embed.shape[0]
        ps = self.estimator(obs_embed)
        phys_losses = {}

        if self._prev is not None and self._prev.B == B:
            Fext = self.F_head(a).reshape(B, self.N, 3)
            tau  = self.tau_head(a).reshape(B, self.N, 3)
            pred = self.prior_ode(self._prev, Fext, tau)
            corr = self.corrector(pred.flatten(), obs_embed)
            corrected = pred.flatten() + corr
            pl, phys_losses = self.laws.total(self._prev, ps, Fext, tau)
            phys_losses["physics/total_loss"]    = pl
            phys_losses["physics/residual_norm"] = corr.norm().item()
            embed = self.phys_proj(corrected)
        else:
            embed = self.phys_proj(ps.flatten())

        self._prev = ps
        h, z, pp, qp = self.rssm.observe_step(h, z, a, embed)
        return h, z, pp, qp, ps, phys_losses

    def reset_episode(self):
        self._prev = None
