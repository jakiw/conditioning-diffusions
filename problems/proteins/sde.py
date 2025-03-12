import jax.numpy as jnp
from problems.proteins.system import System
from sdes.sdes import SDE
from helpers import vmap_sde_dimension

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

A = jnp.hstack([system.A, jnp.zeros_like(system.A)], dtype=jnp.float32)
B = jnp.hstack([system.B, jnp.zeros_like(system.B)], dtype=jnp.float32)


def drift(t, x):
    q = x[:ndim]
    v = x[ndim:]
    dq = v
    dv = system.dUdx(q) / system.mass - v * gamma
    ret = jnp.hstack([
        dq,
        dv
    ], dtype=jnp.float32)
    return ret

def sigma(t, x, dBt):
    return xi_sqrt * dBt

def cov(t, x, v):
    print("cov")
    print(xi.dtype)
    print(v.dtype)
    return xi * v

def sigma_transp_inv(t, x, dBt):
    return dBt / xi_sqrt


ada_sde = SDE(
    drift=drift,
    sigma=sigma,
    covariance=cov,
    sigma_transp_inv=sigma_transp_inv
)


def protein_problem():
    y_obs = system.B
    sde = ada_sde
    control = None

    return ada_sde, control, B, A, f"Protein Problem"
