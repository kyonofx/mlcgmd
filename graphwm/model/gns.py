import torch
from torch.distributions.normal import Normal

import graphwm.model.networks as nets
from graphwm.model.base import GraphSim

class GNS(GraphSim):
  """
  graph network simulation
  """  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.soft_plus = torch.nn.Softplus()
    self.dynamics_gn = nets.EncodeProcessDecode(
          node_dim=self.hparams.dimension*(self.hparams.seq_len - 1) + self.latent_dim + int(self.hparams.use_weights),
          edge_dim=self.hparams.dimension+1+self.hparams.type_emb_size,
          out_dim=self.hparams.dimension*2, 
          **self.hparams.dynamics_gn_hparams)
  
  def predict(self, pos_seq, n_node, ptype_embeddings, bonds, weights, 
              lattices=None, deterministic=False):
    """
    predict *next* position and *current* prop.
    """
    input_graph = self._dynamics_preprocessor(pos_seq, ptype_embeddings, n_node, 
                                              bonds, lattices, weights=weights)
    acc_pred, latent_graph = self.dynamics_gn(input_graph, return_latent=True)
    acc_mean, acc_std = torch.split(acc_pred, self.hparams.dimension, dim=-1)
    acc_std = self.hparams.min_std + self.soft_plus(acc_std)
    acc_dist = Normal(acc_mean, acc_std)
    
    if deterministic:
      acc_pred = acc_mean
    else:
      acc_pred = acc_dist.sample()
    next_pos_pred = self._decoder_postprocessor(acc_pred, pos_seq)
    
    if self.hparams.property_net_hparams:
      prop_pred = self.predict_prop(latent_graph, pos_seq[:, -1], weights, n_node)
    else:
      prop_pred = None
      
    return next_pos_pred, prop_pred
 
  def forward(self, pos_seq, next_pos, ptypes, n_node, 
              bonds=None, bond_types=None, labels=None,
              lattices=None, cg_bonds=None, n_cg_node=None, cluster=None, keypoint=None):
    if self.particle_masses.device != pos_seq.device:
      self.particle_masses = self.particle_masses.to(pos_seq.device)
      self.noise_sigmas = self.noise_sigmas.to(pos_seq.device)
    
    weights = self.particle_masses[ptypes]
    ptype_embeddings = self._embedding_preprocessor(ptypes, n_node, bonds, bond_types, weights=weights)
    
    # coarse graining.
    if self.use_coarse_graining:
      (pos_seq, next_pos, ptype_embeddings, weights) = self.coarse_graining(
          pos_seq, next_pos, ptype_embeddings, weights, cluster, keypoint)
      n_node = n_cg_node
      bonds = cg_bonds  # only operate at coarse-level after this point.
    
    pos_seq, next_pos = self.noise_augment(pos_seq, next_pos, n_node)
    input_graph = self._dynamics_preprocessor(pos_seq, ptype_embeddings, n_node, bonds, lattices, weights=weights)
    
    # compute acceleration loss.
    acc_stats, latent_graph = self.dynamics_gn(input_graph, return_latent=True)
    acc_mean, acc_std = torch.split(acc_stats, self.hparams.dimension, dim=-1)
    acc_std = self.hparams.min_std + self.soft_plus(acc_std)
    
    acc_dist = Normal(acc_mean, acc_std)
    acc_target = self._inverse_decoder_postprocessor(next_pos, pos_seq)
    acc_loss = -acc_dist.log_prob(acc_target).mean()
    
    # fit rgs residual and compute loss.
    if self.hparams.property_net_hparams:
      prop_pred = self.predict_prop(latent_graph, pos_seq[:, -1], weights, n_cg_node)
      property_loss = (prop_pred - labels[:, -1][:, None]).pow(2).mean()
    else:
      property_loss = 0
    loss = acc_loss + property_loss
    
    return {'loss': loss, 
            'acc_loss': acc_loss,
            'property_loss': property_loss,
            'mean_std_ratio': (acc_mean.norm(dim=-1) / acc_std.norm(dim=-1)).mean(),
            'entropy': acc_dist.entropy().mean()}
      
  def step(self, batch, batch_idx):
    return self(batch['position'], batch['target'], batch['particle_types'],
                batch['n_particle'], batch['bonds'], batch.get('bond_types'), 
                batch.get('rgs'), batch.get('lattices'),
                batch.get('cg_bonds'), batch.get('n_keypoint'), batch.get('cluster'), 
                batch.get('keypoint'))
    
  def _decoder_postprocessor(self, acceleration, pos_seq):
    """apply <acceleration> to <pos_seq> to get next position."""
    most_recent_pos = pos_seq[:, -1]
    most_recent_velocity = most_recent_pos - pos_seq[:, -2]
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_pos = most_recent_pos + new_velocity  # * dt = 1
    return new_pos
  
  def _inverse_decoder_postprocessor(self, next_pos, pos_seq):
    """Inverse of `_decoder_postprocessor`."""
    previous_pos = pos_seq[:, -1]
    previous_velocity = previous_pos - pos_seq[:, -2]
    next_velocity = next_pos - previous_pos
    acceleration = next_velocity - previous_velocity
    return acceleration