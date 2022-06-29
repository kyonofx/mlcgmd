# some utils are from VGPL/DPI.
import os
import json
import numpy as np
import torch
import h5py

from torch_geometric.nn import MessagePassing

class ConnectedComponents(MessagePassing):

  def __init__(self):
    super(ConnectedComponents, self).__init__(aggr="max")
      
  def forward(self, n_node, edge_index):
    x = torch.arange(n_node).view(-1, 1)
    last_x = torch.zeros_like(x)

    while not x.equal(last_x):
      last_x = x.clone()
      x = self.propagate(edge_index, x=x)
      x = torch.max(x, last_x)

    unique, perm = torch.unique(x, return_inverse=True)
    perm = perm.view(-1)
    return unique, perm

  def message(self, x_j):
    return x_j

  def update(self, aggr_out):
    return aggr_out
  

def store_data(data_names, data, path):
  hf = h5py.File(path, 'w')
  for i in range(len(data_names)):
    hf.create_dataset(data_names[i], data=data[i])
  hf.close()

def load_data(data_names, path):
  hf = h5py.File(path, 'r')
  data = []
  for i in range(len(data_names)):
    d = np.array(hf.get(data_names[i]))
    data.append(d)
  hf.close()
  return data

def load_data_w_idx(data_names, path, select_idx):
  hf = h5py.File(path, 'r')
  data = []
  for i in range(len(data_names)):
    d = np.array(hf.get(data_names[i])[select_idx])
    data.append(d)
  hf.close()
  return data

def read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def dict_collate_fn(batch):
  batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
  
  # bonds add bonds offsets
  if 'bonds' in batch.keys():
    index_offsets = torch.cumsum(batch['n_bond'], -1)
    node_offsets = torch.cumsum(batch['n_particle'], -1)
    for idx in range(len(index_offsets)-1):
      batch['bonds'][index_offsets[idx]:index_offsets[idx+1]] += node_offsets[idx]
  
  # GROUPING: add cluster offsets
  if 'n_keypoint' in batch:
    if 'n_component' in batch:
      component_offsets = torch.cumsum(batch['n_component'], -1)
    index_offsets = torch.cumsum(batch['n_keypoint'], -1)
    node_offsets = torch.cumsum(batch['n_particle'], -1)
    bond_offset = torch.cumsum(batch['n_cg_bond'], -1)
    for idx in range(len(index_offsets)-1):
      if 'n_component' in batch:
        batch['component'][node_offsets[idx]:node_offsets[idx+1]] += component_offsets[idx]
      if 'keypoint' in batch:
        batch['keypoint'][index_offsets[idx]:index_offsets[idx+1]] += node_offsets[idx]
      batch['cluster'][node_offsets[idx]:node_offsets[idx+1]] += index_offsets[idx]
      batch['cg_bonds'][bond_offset[idx]:bond_offset[idx+1]] += index_offsets[idx]    
  return batch