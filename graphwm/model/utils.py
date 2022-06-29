import torch
from torch_scatter import scatter
from torch_cluster import radius_graph

EPSILON = 1e-8

PSEUDO_MASSES = [1.] * 100

ATOM_MASSES = [
    0.0, 1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999,
    18.998403163, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955908,
    47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62,
    88.90584, 91.224, 92.90637, 95.95, 97.90721, 101.07, 102.9055, 106.42,
    107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 144.91276,
    150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.045, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217,
    195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209.0, 210.0,
    222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588, 238.02891, 237.0, 244.0,
    243.0, 247.0, 247.0, 251.0, 252.0]

def get_position_noise(position_sequence, noise_std, use_rw=False):
  """
  Returns random-walk noise in the velocity applied to the position.
  For systems with PBC, takes in the unwrapped coordinates.
  """
  if use_rw:
    velocity_sequence = position_sequence[:, 1:] - position_sequence[:, :-1]
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = (torch.randn_like(velocity_sequence) * 
                              (noise_std / num_velocities ** 0.5))
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)
  else:
    position_sequence_noise = torch.randn_like(position_sequence) * noise_std

  return position_sequence_noise

def get_position_noise_with_velocity(position_sequence, noise_std, use_rw=False):
  """
  Returns random-walk noise in the velocity applied to the position.
  For systems with PBC, takes in the unwrapped coordinates.
  now add noise to velocity.
  """
  velocity_sequence = position_sequence[:, 1:] - position_sequence[:, :-1]
  
  if use_rw:
    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = (torch.randn_like(velocity_sequence) * 
                              (noise_std / num_velocities ** 0.5))
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
  else:
    velocity_sequence_noise = torch.rand_like(velocity_sequence) * noise_std
  # Integrate the noise in the velocity to the positions, assuming
  # an Euler intergrator and a dt = 1, and adding no noise to the very first
  # position (since that will only be used to calculate the first position
  # change).
  position_sequence_noise = torch.cat([
      torch.zeros_like(velocity_sequence_noise[:, 0:1]),
      torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

  return position_sequence_noise

def compute_weighted_rgs(positions, weights, n_node):
  """
  Inputs:
      positions: N x dim
      weights: N x 1
      n_node: B x 1
  Outputs:
      rgs: B x 1
  """
  positions_per_graph_list = torch.split(positions, list(n_node), dim=0)
  weights_per_graph_list = torch.split(weights, list(n_node), dim=0)
  all_rgs = []
  for positions, weights in zip(positions_per_graph_list, weights_per_graph_list):
    com = (positions * weights).sum(0) / weights.sum()
    rgs = (weights.squeeze()*((positions - com).pow(2)).sum(1)).sum() / weights.sum()
    rgs = rgs.sqrt()
    all_rgs.append(rgs.view(-1))
  return torch.cat(all_rgs).unsqueeze(1)

def get_n_edge(senders, n_node):
  """
  return number of edges for each graph in the batched graph. 
  Has the same shape as <n_node>.
  """
  index_offsets = torch.cat([torch.zeros(1).to(n_node.device), 
                             torch.cumsum(n_node, -1)], dim=-1)
  n_edge = torch.LongTensor([torch.logical_and(senders >= index_offsets[i], 
                                               senders < index_offsets[i+1]).sum() 
                             for i in range(len(n_node))])
  return n_edge

def compute_connectivity(positions, n_node, radius, bonds=None, add_self_edges=True):
  batch = torch.arange(len(n_node)).to(n_node.device).repeat_interleave(n_node, dim=0)
  senders, receivers = radius_graph(positions, radius, batch, loop=add_self_edges)
  if bonds is not None:
    edge_indices = torch.cat([senders.unsqueeze(1), receivers.unsqueeze(1)], dim=1)
    all_edges = torch.cat([edge_indices, bonds], dim=0)
    all_edges, counts = torch.unique(all_edges, dim=0, return_counts=True)
    edge_types = (counts > 1).int()
    senders, receivers = all_edges[:, 0], all_edges[:, 1]
  else:
    edge_types = None
  # displacements normalized with radius.
  displacements = (positions[senders] - positions[receivers]) / radius
  distances = displacements.norm(dim=-1, keepdim=True)
  return senders, receivers, displacements, distances, edge_types

# Below: utils for systems with periodic boundaru conditions.

def compute_connectivity_pbc(positions, lattices, n_node, radius, 
                             bonds=None, add_self_edges=True):
  """
  construct graph using radius cut-off under periodic boundary conditions.
  simplified computation of distances/displacements under PBC is possible
  due to the lattices being cubes.
  Inputs:
    positions: N x dim
    lattices: B x dim
    n_node: B x 1
  Optional Inputs:
    bonds: E x 2. include these bonds even if they don't fall into the radius cut-off.
  Outputs:
    senders, receivers, displacements, distances, edge_type (1 if is_bond)
  """
  n_node_per_image_sqr = (n_node ** 2).long()

  # index offset between images
  index_offset = (torch.cumsum(n_node, dim=0) - n_node)
  index_offset_expand = torch.repeat_interleave(index_offset, n_node_per_image_sqr)
  n_node_per_image_expand = torch.repeat_interleave(n_node, n_node_per_image_sqr)

  # Compute a tensor containing sequences of numbers that range 
  # from 0 to n_node_per_image_sqr for each image
  # that is used to compute indices for the pairs of atoms.
  num_atom_pairs = torch.sum(n_node_per_image_sqr)
  index_sqr_offset = (torch.cumsum(n_node_per_image_sqr, dim=0) - n_node_per_image_sqr)
  index_sqr_offset = torch.repeat_interleave(index_sqr_offset, n_node_per_image_sqr)
  atom_count_sqr = (torch.arange(num_atom_pairs, device=positions.device) - index_sqr_offset)

  # Compute the indices for the pairs of atoms (using division and mod)
  index1 = ((atom_count_sqr.div(n_node_per_image_expand, rounding_mode='floor'))).long() + index_offset_expand
  index2 = (atom_count_sqr % n_node_per_image_expand).long() + index_offset_expand
  # Get the positions for each atom
  pos1 = torch.index_select(positions, 0, index1)
  pos2 = torch.index_select(positions, 0, index2)

  lattices = lattices.repeat_interleave(n_node_per_image_sqr, dim=0)
  displacement = pos1 - pos2  
  displacement = torch.where(displacement > (0.5 * lattices),
                             displacement - lattices, displacement)
  displacement = torch.where(displacement < (-0.5 * lattices),
                             displacement + lattices, displacement)
  displacement = displacement / radius
  distance = displacement.norm(dim=-1)

  mask = torch.le(distance, 1)
  if not add_self_edges:
    mask_not_same = torch.gt(distance, EPSILON)
    mask = torch.logical_and(mask, mask_not_same)
  
  if bonds is not None:
    all_edges = torch.cat([index1[:, None], index2[:, None]], dim=1)
    all_edges_non_unique = torch.cat([all_edges, bonds], dim=0)
    all_edges, counts = torch.unique(all_edges_non_unique, dim=0, return_counts=True)
    bond_mask = (counts > 1)  # this is correct because <all_edges> enumerates all edges.
    mask = torch.logical_or(mask, bond_mask)  # include both raidus-edges and bonds.
    # get edge type.
    all_edge_indices, counts = torch.unique(
        torch.cat([mask.nonzero(), bond_mask.nonzero()], dim=0), return_counts=True)
    edge_types = (counts > 1).int()
  else:
    edge_types = None
  # senders, receivers, displacements, distances, edge_type (is_bond)
  return (index1[mask], index2[mask], displacement[mask], 
          distance[mask].unsqueeze(1), edge_types)

def wrap_positions(u_positions, lattices): 
  """
  Inputs:
    u_positions: N x T x dim
    lattices: N x dim
  Return:
    positions: N x T x dim
  """
  return (((u_positions.transpose(0, 1) / lattices) % 1) * lattices).transpose(0, 1)

def distance_pbc(x0, x1, lattices):
  """
  distance between atoms in periodic boundary conditions.
  """
  delta = torch.abs(x0 - x1)
  delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
  return torch.sqrt((delta ** 2).sum(axis=-1))
  
def displacement_pbc(x0, x1, lattices):
  """
  The coordinate of atoms in x1 relative to x0 in periodic boundary conditions.
  """
  delta = x1 - x0
  delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
  delta = torch.where(delta < -0.5 * lattices, delta + lattices, delta)
  return delta

def compute_com(positions, weights, cluster):
  """
  compute center of mass according to the cluster assignments.
  Inputs:
    positions: N x T x dim
    weights: N x 1
    cluster: N x 1
  """
  weights = weights.view(-1, 1, 1)  
  cg_weights = scatter(weights.squeeze(), cluster, dim=0, reduce='sum').view(-1, 1, 1)
  cg_positions = scatter(weights*positions, cluster, dim=0, reduce='sum') / cg_weights
  cg_weights = cg_weights.view(-1, 1)
  return cg_positions, cg_weights