import jax.numpy as jnp
from problems.proteins.system import System
from sdes.sdes import SDE
from helpers import vmap_sde_dimension
from metrics.metrics import get_energy_metric

temperature = 300 # from configs/aldp_diagonal_single_gaussian
gamma = 1.0

start = "problems/proteins/files/AD_A.pdb" # from configs/aldp_diagonal_single_gaussian
target = "problems/proteins/files/AD_B.pdb" # from configs/aldp_diagonal_single_gaussian
force_clip = 1e7 # from configs/aldp_diagonal_single_gaussian
forcefield = ['problems/proteins/amber14/protein.ff14SB.xml', 'problems/proteins/amber14/tip3p.xml'] # from main.py
cv = 'phi_psi' # from main.py

system = System.from_pdb(start, target, forcefield, cv, force_clip)

kbT = 1.380649 * 6.02214076 * 1e-3 * temperature
xi_vel = jnp.sqrt(2 * kbT * gamma / system.mass)

xi_pos_noise = 1e-4 # from main.py
xi_pos = jnp.ones_like(xi_vel) * xi_pos_noise

xi = jnp.concatenate((xi_pos, xi_vel), axis=-1, dtype=jnp.float32)
xi_sqrt = xi**0.5

ndim = system.A.shape[0]

def drift(t, x):
    dx = -system.dUdx(x) / (gamma * system.mass)
    return dx

def sigma(t, x, dBt):
    return xi_vel**0.5 * dBt

def cov(t, x, v):
    return xi_vel * v

def sigma_transp_inv(t, x, dBt):
    return dBt / xi_vel**0.5


ada_sde_first_order = SDE(
    drift=drift,
    sigma=sigma,
    covariance=cov,
    sigma_transp_inv=sigma_transp_inv
)

def protein_problem(ts):
    sde = ada_sde_first_order
    energy = system.U
    energy_metric = get_energy_metric(sde, energy, ts)

    return sde, {"energy": energy_metric}, system.B, system.A, f"Proteins"



A = jnp.hstack([system.A, jnp.zeros_like(system.A)], dtype=jnp.float32)
B = jnp.hstack([system.B, jnp.zeros_like(system.B)], dtype=jnp.float32)

def drift_so(t, x):
    q = x[:ndim]
    v = x[ndim:]
    dq = v
    dv = -system.dUdx(q) / system.mass - v * gamma
    ret = jnp.hstack([
        dq,
        dv
    ], dtype=jnp.float32)
    return ret

def sigma_so(t, x, dBt):
    return xi_sqrt * dBt

def cov_so(t, x, v):
    return xi * v

def sigma_transp_inv_so(t, x, dBt):
    return dBt / xi_sqrt


ada_sde_second_order = SDE(
    drift=drift_so,
    sigma=sigma_so,
    covariance=cov_so,
    sigma_transp_inv=sigma_transp_inv_so
)




def protein_problem_so(ts):
    y_obs = system.B
    sde = ada_sde_second_order

    def energy(x):
        q = x[:ndim]
        return system.U(q)

    energy_metric = get_energy_metric(sde, energy, ts)

    return sde, {"energy": energy_metric}, B, A, f"Proteins Second Order"
