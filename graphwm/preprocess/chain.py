import numpy as np
import os

import mdtraj

ATOM_TYPES = ['B1', 'B2', 'S1', 'S2']

def read_rg(rg_file):
    with open(rg_file) as f:
        rows = [row for row in f]
    rgs = [float(row.split()[1]) for row in rows[2:]]
    return np.array(rgs)
  
def load_polymer_rg(data_dir):
    """
    atom_coords: float, (num_traj, num_atom, 3)
    atom_types: int, (num_atom,)
    lattices: float, (num_traj, 3)
    bond_indices: int, (num_bonds, 2)
    rgs: float, (num_traj,)
    """
    traj = mdtraj.load(os.path.join(data_dir, 'coords.lammpstrj'),
                       top=os.path.join(data_dir, 'poly-0.pdb'))
    # scale units to A
    atom_coords = traj.xyz * 10.
    lattices = traj.unitcell_lengths * 10.

    # get topology of the graph
    table, bonds = traj.top.to_dataframe()

    atom_types = [ATOM_TYPES.index(atom) for atom in table.name]
    atom_types = np.array(atom_types)

    # shape (num_bonds, 2), denotes indices of atoms connected by bonds
    bond_indices = bonds[:, :2]
    rgs = read_rg(os.path.join(data_dir, 'rg.txt'))

    assert atom_coords.shape[0] == lattices.shape[0] == rgs.shape[0]
    import pdb; pdb.set_trace()
    return [atom_coords, lattices, rgs, atom_types, bond_indices]