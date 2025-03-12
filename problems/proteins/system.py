import openmm.app as app
import openmm.unit as unit
import jax.numpy as jnp
import jax
from dmff import Hamiltonian, NeighborList  # This sets jax to use 64-bit precision
from typing import Callable, Optional, Self
from jax.typing import ArrayLike



# This code is https://github.com/plainerman/Variational-Doob

def assert_same_molecule(A: app.PDBFile, B: app.PDBFile):
    """Check whether the two PDB files are equal (up to atom positions)."""
    assert A.topology.getNumChains() == B.topology.getNumChains(), "Number of chains do not match"

    # Compare chains, residues, and atoms
    for chainA, chainB in zip(A.topology.chains(), B.topology.chains()):
        assert len(list(chainA.residues())) == len(list(chainB.residues())), "Number of residues do not match"

        for residueA, residueB in zip(chainA.residues(), chainB.residues()):
            assert len(list(residueA.atoms())) == len(list(residueB.atoms())), "Number of atoms do not match"

            assert [a.element for a in residueA.atoms()] == [a.element for a in
                                                             residueB.atoms()], "Elements do not match"

    assert A.topology.getNumBonds() == B.topology.getNumBonds(), "Number of bonds do not match"
    for bondA, bondB in zip(A.topology.bonds(), B.topology.bonds()):
        assert bondA[0].element == bondB[0].element, "Elements of bond atoms do not match"
        assert bondA[1].element == bondB[1].element, "Elements of bond atoms do not match"


@jax.jit
def kabsch_align(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: Return aligned P and Q
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = jnp.dot(p.T, q)

    # SVD
    U, _, Vt = jnp.linalg.svd(H)

    # Validate right-handed coordinate system
    det = jnp.linalg.det(jnp.dot(Vt.T, U.T))
    Vt = jnp.where(det < 0.0, Vt.at[-1, :].set(Vt[-1, :] * -1.0), Vt)

    # Optimal rotation
    R = jnp.dot(Vt.T, U.T)

    return jnp.dot(p, R.T), q


@jax.jit
def kabsch_rmsd(P, Q):
    P_aligned, Q_aligned = kabsch_align(P, Q)
    return jnp.sqrt(jnp.sum(jnp.square(P_aligned - Q_aligned)) / P.shape[0])


class System:
    def __init__(self, U: Callable[[ArrayLike], ArrayLike], A: ArrayLike, B: ArrayLike, mass: ArrayLike,
                 force_clip: float):
        assert A.shape == B.shape == mass.shape

        self.U = jax.jit(U)

        dUdx = jax.grad(lambda _x: U(_x).sum())
        self.dUdx = jax.jit(lambda _x: jnp.clip(dUdx(_x), -force_clip, force_clip))

        self.A, self.B = A, B
        self.mass = mass

    @classmethod
    def from_pdb(cls, A: str, B: str, forcefield: [str], cv: Optional[str], force_clip: float) -> Self:
        A_pdb, B_pdb = app.PDBFile(A), app.PDBFile(B)
        assert_same_molecule(A_pdb, B_pdb)

        mass = [a.element.mass.value_in_unit(unit.dalton) for a in A_pdb.topology.atoms()]
        mass = jnp.broadcast_to(jnp.array(mass, dtype=jnp.float32).reshape(-1, 1), (len(mass), 3)).reshape(-1)

        A = jnp.array(A_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=jnp.float32)
        B = jnp.array(B_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=jnp.float32)
        num_atoms = A.shape[0]
        A, B = kabsch_align(A, B)
        A, B = A.reshape(-1), B.reshape(-1)

        # Initialize the potential energy with amber forcefields
        ff = Hamiltonian(*forcefield)
        potentials = ff.createPotential(A_pdb.topology,
                                        nonbondedMethod=app.NoCutoff,
                                        nonbondedCutoff=1.0 * unit.nanometers,
                                        constraints=None,
                                        ewaldErrorTolerance=0.0005)

        # Create a box used when calling
        box = jnp.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
        nbList = NeighborList(box, 4.0, potentials.meta["cov_map"])
        nbList.allocate(A.reshape(-1, 3))

        _U = potentials.getPotentialFunc()

        def U(_x):
            return _U(_x.reshape(num_atoms, 3), box, nbList.pairs, ff.paramset.parameters).sum()

        # if cv is None:
        #     plot = None
        # elif cv == 'phi_psi':
        #     mdtraj_topology = md.Topology.from_openmm(A_pdb.topology)
        #     phis_psis = phi_psi_from_mdtraj(mdtraj_topology)

        #     from utils.plot import plot_cv
        #     plot = partial(plot_cv,
        #                     cv=phis_psis,
        #                     bins=100, states=list(zip(['A', 'B'], [phis_psis(A[None]), phis_psis(B[None])])),
        #                     xlim=jnp.array((-jnp.pi, jnp.pi)), ylim=jnp.array((-jnp.pi, jnp.pi)),
        #                     xlabel=r'$\phi$', ylabel=r'$\psi',
        #                     xticks=[-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        #                     xticklabels=[r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'],
        #                     yticks=[-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        #                     yticklabels=[r'$-\pi$', r'$-\frac {\pi} {2}$', '0', r'$\frac {\pi} {2}$', r'$\pi$'],
        #                     square=True, periodic=True,
        #                     )
        # else:
        #     raise ValueError(f"Unknown cv: {cv}")

        return cls(U, A, B, mass, force_clip)