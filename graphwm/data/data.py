import numpy as np
import torch
from pathlib import Path

from torch.utils.data import Dataset
from torch_scatter import scatter

import graphwm.data.utils as utils
from graphwm.data.clustering import metis_cluster

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

class MDDataset(Dataset):
  """
  Args:
    directory:         directory to data.
    split:             subset of data to use.
    seq_len:           number of frames of each sampled data point.
    dilation:          the time interval between each two sampled frame. 
    grouping:          every <grouping> fine-level particles will be grouped into one coarse-level bead.
    traj_len:          length of the whole trajectory to be loaded.
    recursive_metis:   if <True>, use recursive METIS clustering.
  """
  def __init__(self, 
               directory, 
               split, 
               mode='train',
               seq_len=1, 
               dilation=1, 
               grouping=1,
               traj_len=100000,
               recursive_metis=False):
    if isinstance(directory, str):
      directory = Path(directory)
    self.directory = directory
    
    assert mode in ['rollout', 'train', 'oneshot']
    self.mode = mode
    
    self.seq_len = seq_len
    if self.mode == 'train':
      # +1 for count the target pos
      self.seq_len += 1
    self.grouping = grouping
    self.recursive_metis = recursive_metis
    self.dilation = dilation
        
    with open(split, 'r') as f:
      traj_index = f.read().splitlines()
    self.traj_index = sorted([self.directory / traj for traj in traj_index])
    self.n_rollouts = len(self.traj_index)
    
    self.traj_len = traj_len
    self.offset = self.traj_len - self.seq_len*self.dilation + 1
  
  def get_file_indices(self, idx):
    if self.mode == 'rollout':
      idx_rollout = idx
      st_idx = 0
      ed_idx = st_idx + self.seq_len * self.dilation
    elif self.mode == 'train':
      idx_rollout = idx // self.offset
      st_idx = idx % self.offset
      ed_idx = st_idx + self.seq_len * self.dilation
    elif self.mode == 'oneshot':
      # oneshot model only uses the particle types.
      idx_rollout = idx
      st_idx = 0
      ed_idx = st_idx + self.traj_len
    return idx_rollout, st_idx, ed_idx
  
  def get_cg_stats(self, tensor_dict):
    num_cg_nodes = round(int(tensor_dict['n_particle']) / self.grouping)
    row, col = tensor_dict['bonds'][:, 0], tensor_dict['bonds'][:, 1]
    cluster, keypoints = metis_cluster(num_cg_nodes, tensor_dict['n_particle'], 
                                      row, col, recursive=self.recursive_metis)
    boundary_bonds = (cluster[row] != cluster[col])
    cg_bonds = torch.cat([cluster[row[boundary_bonds]][:, None], 
                          cluster[col[boundary_bonds]][:, None]], dim=1)
    cg_bonds = torch.unique(cg_bonds, dim=0)
    
    return {
        'keypoint': keypoints,
        'n_keypoint': torch.LongTensor([num_cg_nodes]),
        'cluster': cluster,
        'cg_bonds': cg_bonds,
        'n_cg_bond': torch.LongTensor([len(cg_bonds)])
          }
      
  def __len__(self):
    if self.mode in ['rollout', 'oneshot']:
      return self.n_rollouts
    else:
      return self.n_rollouts * self.offset
    
  def __getitem__(self, idx):
    raise NotImplementedError
  
class PolymerDataset(MDDataset):
  """
  Fields:
    position: T x N x 3
    rgs: T x 1
    bond_indices: (N-1) x 2
    particle_types: N x 1
  """
    
  def __getitem__(self, idx):
    idx_rollout, st_idx, ed_idx = self.get_file_indices(idx)
    select_idx = np.arange(st_idx, ed_idx, self.dilation, dtype=np.int64)
    tensor_dict = {}
    
    ptype = utils.load_data(['particle_type'], self.traj_index[idx_rollout] / 'ptype.h5')[0]
    rgs = utils.load_data_w_idx(['rgs'], self.traj_index[idx_rollout] / 'rgs.h5', select_idx)[0]
    bonds = utils.load_data(['bond_indices'], self.traj_index[idx_rollout] / 'bond.h5')[0]
      
    reversed_bonds = np.concatenate([bonds[:, 1:], bonds[:, :1]], axis=1)
    bonds = np.concatenate([bonds, reversed_bonds], axis=0)
    tensor_dict.update({'bonds': np.array(bonds, dtype=np.int64),
                        'n_bond': np.array([bonds.shape[0]], dtype=np.int64)})
    
    if self.mode == 'oneshot':
      tensor_dict.update({'particle_types': np.array(ptype, dtype=np.int64),
                          'rgs': np.array(rgs, dtype=np.float32)[None, :],
                          'n_particle': np.array([ptype.shape[0]], dtype=np.int64)})
      tensor_dict = {k: torch.from_numpy(v) for k, v in tensor_dict.items()}   
      return tensor_dict
    
    positions = utils.load_data_w_idx(['position'], self.traj_index[idx_rollout] / 'position.h5', 
                                      select_idx)[0]
    if self.mode == 'train':
      input_pos = positions[:-1]  
      input_rgs = rgs[:-1]
    else:
      input_pos = positions
      input_rgs = rgs
    target_pos = positions[-1]
    target_rgs = rgs[-1]

    tensor_dict.update({
        'position': np.array(input_pos, dtype=np.float32).transpose(1, 0, 2),  # N x T x dim
        'rgs': np.array(input_rgs, dtype=np.float32)[None, :],
        'target': np.array(target_pos, dtype=np.float32),
        'target_rgs': np.array([target_rgs], dtype=np.float32)[None, :],
        'particle_types': np.array(ptype, dtype=np.int64),
        'n_particle': np.array([target_pos.shape[0]], dtype=np.int64)
    })
    
    # clustering use torch tensors.
    tensor_dict = {k: torch.from_numpy(v) for k, v in tensor_dict.items()}
    tensor_dict.update(self.get_cg_stats(tensor_dict))
        
    return tensor_dict

class BatteryDataset(MDDataset):

  def __init__(self, directory, split, 
               mode='train', seq_len=1, dilation=1, grouping=1,
               traj_len=2500, recursive_metis=True, cg_tfsi=False, remove_com=False):
    """
    cg_tfsi:    if <True> always group each TFSI molecule to a single coarse-grained bead.
    remove_com: if <True> remove drifting by removing the CoM of ths system from each time step.
    """
    self.cg_tfsi = cg_tfsi
    self.remove_com = remove_com
    self.atom_masses = np.array(ATOM_MASSES)
    self.connected_components = utils.ConnectedComponents()
    
    super().__init__(directory, split, mode, seq_len, dilation, 
                     grouping, traj_len, recursive_metis)
    
  def get_cg_stats(self, tensor_dict):
    """
    cluster each polymer chain separately by checking the connected components in the bond graph.
    
    if *<cg_level> >= 15*, METIS is not applied to TFSI particles.
    Each TFSI particle will be grouped into a single coarse-grained particle.
    """
    end_points, perm = self.connected_components(int(tensor_dict['n_particle']), tensor_dict['bonds'].T)
    if self.grouping >= 15 or self.cg_tfsi:
      poly_mask = scatter(torch.ones(tensor_dict['n_particle']), perm, reduce='sum') > 15  # 15 is the size of a TFSI
    else:
      poly_mask = scatter(torch.ones(tensor_dict['n_particle']), perm, reduce='sum') > 1
    poly_end_points = torch.cat([torch.zeros(1).long(), end_points[poly_mask]+1])  # +1: the end_points belong to the previous cluster.
    row, col = tensor_dict['bonds'][:, 0], tensor_dict['bonds'][:, 1]
    
    num_cg_nodes = 0
    cluster = []
    for idx in range(len(poly_end_points)-1):
      this_poly_n_particle = int(poly_end_points[idx+1] - poly_end_points[idx])
      
      this_poly_bond_indices = torch.logical_and(row >= poly_end_points[idx], 
                                                row < poly_end_points[idx+1])
      this_poly_n_cg_node = round((int(this_poly_n_particle) / self.grouping))
      this_cluster, _ = metis_cluster(this_poly_n_cg_node, this_poly_n_particle,
                                      row[this_poly_bond_indices]-poly_end_points[idx], 
                                      col[this_poly_bond_indices]-poly_end_points[idx],
                                      recursive=True)
      this_cluster += num_cg_nodes
      num_cg_nodes += this_poly_n_cg_node
      cluster.append(this_cluster)
          
    cluster = torch.cat(cluster)
    # if cg_level >= 15, manually construct TFSI cluster assignments.
    if self.grouping >= 15 or self.cg_tfsi:
      TFSI_clusters = num_cg_nodes + torch.arange(50).repeat_interleave(15)
      cluster = torch.cat([cluster, TFSI_clusters], dim=0)
      num_cg_nodes += 50
    
    # Li cluster assginment. each Li ion belongs to its own cluster.
    Li_clusters = num_cg_nodes + torch.arange(50)
    cluster = torch.cat([cluster, Li_clusters], dim=0)
    num_cg_nodes += 50
    
    # cg bonds.
    boundary_bonds = (cluster[row] != cluster[col])
    cg_bonds = torch.cat([cluster[row[boundary_bonds]][:, None], 
                          cluster[col[boundary_bonds]][:, None]], dim=1)
    cg_bonds = torch.unique(cg_bonds, dim=0)
    
    # component.
    component = torch.cat([end_points[0:1]+1, end_points[1:] - end_points[:-1]])
    component = torch.arange(len(end_points)).repeat_interleave(component)
    
    cg_stats = {
        'component': component,  # annotate the connected component that a particle belongs to.
        'n_keypoint': torch.LongTensor([num_cg_nodes]),
        'n_component': torch.LongTensor([max(component)+1]),
        'cluster': cluster,
        'cg_bonds': cg_bonds,
        'n_cg_bond': torch.LongTensor([len(cg_bonds)])
          }
      
    return cg_stats
  
  def __getitem__(self, idx):
    idx_rollout, st_idx, ed_idx = self.get_file_indices(idx)
    select_idx = np.arange(st_idx, ed_idx, self.dilation, dtype=np.int64)
    tensor_dict = {}

    ptype = utils.load_data(
        ['particle_type'], self.traj_index[idx_rollout] / 'ptype.h5')[0]
    bonds = utils.load_data(
        ['bond_indices'], self.traj_index[idx_rollout] / 'bond.h5')[0]
    bond_types = utils.load_data(
        ['bond_type'], self.traj_index[idx_rollout] / 'bond_type.h5')[0]

    reversed_bonds = np.concatenate([bonds[:, 1:], bonds[:, :1]], axis=1)
    bonds = np.concatenate([bonds, reversed_bonds], axis=0)
    bond_types = np.concatenate([bond_types, bond_types], axis=0)
    tensor_dict.update({'bonds': np.array(bonds, dtype=np.int64),
                        'bond_types': np.array(bond_types, dtype=np.int64),
                        'n_bond': np.array([bonds.shape[0]], dtype=np.int64)})
    
    if self.mode == 'oneshot':
      tensor_dict.update({'particle_types': np.array(ptype, dtype=np.int64),
                          'n_particle': np.array([ptype.shape[0]], dtype=np.int64)})
      tensor_dict = {k: torch.from_numpy(v) for k, v in tensor_dict.items()}   
      return tensor_dict
    
    unwrapped_positions = utils.load_data_w_idx(
        ['unwrapped_position'], self.traj_index[idx_rollout] / 'unwrapped_position.h5', select_idx)[0]
    lattices = utils.load_data_w_idx(
        ['lattice'], self.traj_index[idx_rollout] / 'lattice.h5', select_idx)[0][-1]
    
    if self.remove_com:
      weights = self.atom_masses[ptype]
      CoM = np.sum(unwrapped_positions * weights.reshape(1, -1, 1), axis=1) / np.sum(weights)
      unwrapped_positions = unwrapped_positions - CoM[:, None, :]
      
    if self.mode == 'train':
      input_uw_pos = unwrapped_positions[:-1]
    else:
      input_uw_pos = unwrapped_positions
    target_uw_pos = unwrapped_positions[-1]

    tensor_dict.update({
        'position': np.array(input_uw_pos, dtype=np.float32).transpose(1, 0, 2),  # N x T x dim
        'target': np.array(target_uw_pos, dtype=np.float32),
        'lattices': np.array([lattices], dtype=np.float32),
        'particle_types': np.array(ptype, dtype=np.int64),
        'n_particle': np.array([target_uw_pos.shape[0]], dtype=np.int64),
    })
    
    # clustering use torch tensors.
    tensor_dict = {k: torch.from_numpy(v) for k, v in tensor_dict.items()}
    if self.grouping > 1:
      tensor_dict.update(self.get_cg_stats(tensor_dict))
      
    return tensor_dict