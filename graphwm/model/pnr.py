import math
import torch
from torch.distributions.normal import Normal

import graphwm.model.networks as nets
from graphwm.model.gns import GNS

STD_EPSILON = 1e-8

class PnR(GNS):
  """
  predict then refine. score-based model training adapted from NCSN.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.sigmas = torch.linspace(math.log(self.hparams.sigma_begin), 
                                 math.log(self.hparams.sigma_end), 
                                 self.hparams.sigma_level).exp()
    self.sigmas.requires_grad = False
    
    self.score_gn = nets.EncodeProcessDecode(
          node_dim=self.latent_dim+int(self.hparams.use_weights),
          edge_dim=self.hparams.dimension+1+self.hparams.type_emb_size,
          out_dim=self.hparams.dimension, 
          **self.hparams.score_gn_hparams)

  def predict(self, pos_seq, n_node, ptype_embeddings, bonds, weights, 
              lattices=None, deterministic=False):
    history_graph = self._dynamics_preprocessor(pos_seq, ptype_embeddings, n_node, bonds, lattices, weights=weights)
    acc_pred, history_latent_graph = self.dynamics_gn(history_graph, return_latent=True)
    history_latent = history_latent_graph.nodes
    
    acc_mean, acc_std = torch.split(acc_pred, self.hparams.dimension, dim=-1)
    acc_std = self.hparams.min_std + self.soft_plus(acc_std)
    acc_dist = Normal(acc_mean, acc_std)
    if deterministic:
      acc_pred = acc_mean
    else:
      acc_pred = acc_dist.sample()
    current_pos = self._decoder_postprocessor(acc_pred, pos_seq)
    
    if self.hparams.property_net_hparams:
      prop_pred = self.predict_prop(history_latent_graph, pos_seq[:, -1], weights, n_node)
    else:
      prop_pred = None
    # start reverse diffusion from latest history
    current_pos = current_pos.detach().clone()
    # annealed langevin dynamics.
    for sigma in self.sigmas:  # [-self.hparams.dyn_level:]
      for ld_step in range(self.hparams.step_per_sigma):
        score_graph = self._dynamics_preprocessor(
            current_pos[:, None], history_latent, n_node, bonds, lattices, weights=weights)
        scores = self.hparams.step_size * self.score_gn(score_graph) / sigma
        langevin_noise = torch.randn_like(current_pos) * math.sqrt(self.hparams.step_size * 2)
        current_pos = current_pos + scores + langevin_noise
        
    return current_pos, prop_pred  
  
  def forward(self, pos_seq, next_pos, ptypes, n_node, bonds, 
              bond_types=None, labels=None,
              lattices=None, cg_bonds=None, n_cg_node=None, 
              cluster=None, keypoint=None):
    if self.particle_masses.device != pos_seq.device:
      self.particle_masses = self.particle_masses.to(pos_seq.device)
      self.noise_sigmas = self.noise_sigmas.to(pos_seq.device)
      self.sigmas = self.sigmas.to(pos_seq.device)
    
    weights = self.particle_masses[ptypes]
    ptype_embeddings = self._embedding_preprocessor(ptypes, n_node, bonds, bond_types, weights=weights)
    
    # coarse graining.
    if self.use_coarse_graining:
      (pos_seq, next_pos, ptype_embeddings, weights) = self.coarse_graining(
          pos_seq, next_pos, ptype_embeddings, weights, cluster, keypoint)
      n_node = n_cg_node
      bonds = cg_bonds  # only operate at coarse-level after this point.
    
    pos_seq, next_pos = self.noise_augment(pos_seq, next_pos, n_node)
    sampled_sigmas, noisy_next_pos = self.sample_noisy_pos(next_pos, n_node)
    target = (-1/(sampled_sigmas.pow(2)) * (noisy_next_pos - next_pos)).squeeze()
    # maybe not divide this sigma
    
    history_graph = self._dynamics_preprocessor(
        pos_seq, ptype_embeddings, n_node, bonds, lattices, weights=weights)
    acc_stats, history_latent_graph = self.dynamics_gn(history_graph, return_latent=True)
    acc_mean, acc_std = torch.split(acc_stats, self.hparams.dimension, dim=-1)
    acc_std = self.hparams.min_std + self.soft_plus(acc_std)
    acc_dist = Normal(acc_mean, acc_std)
    acc_target = self._inverse_decoder_postprocessor(next_pos, pos_seq)
    acc_loss = -acc_dist.log_prob(acc_target).mean()
    
    history_latent = history_latent_graph.nodes
    score_graph = self._dynamics_preprocessor(
        noisy_next_pos[:, None], history_latent, n_node, bonds, lattices, weights=weights)
    scores = self.score_gn(score_graph)
    scores = (scores * (1. / sampled_sigmas.view(-1, 1)))
    
    score_loss = (0.5 * ((scores - target) ** 2) * 
                  (sampled_sigmas ** self.hparams.anneal_power)).mean()
    
    # use positions without noise to predict rgs.
    if self.hparams.property_net_hparams:
      prop_pred = self.predict_prop(history_latent_graph, pos_seq[:, -1], weights, n_node)
      property_loss = (prop_pred - labels[:, -1][:, None]).pow(2).mean()
    else:
      property_loss = 0
    
    loss = acc_loss + score_loss + property_loss
    return {'loss': loss, 
            'acc_loss': acc_loss,
            'score_loss': score_loss,
            'property_loss': property_loss}
    
  def sample_noisy_pos(self, pos, n_node):
    noise_level = torch.randint(0, self.hparams.sigma_level, (n_node.shape[0],), 
                                device=n_node.device)
    sampled_sigmas = self.sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1)
    added_noise = sampled_sigmas * torch.randn_like(pos)
    noisy_pos = pos + added_noise
    return sampled_sigmas, noisy_pos
  
  def step(self, batch, batch_idx):
    # If make bond optimal, can hopefully incorporate GNS datasets.
    return self(batch['position'], batch['target'], batch['particle_types'],
                batch['n_particle'], batch['bonds'], 
                batch.get('bond_types'), batch.get('rgs'), batch.get('lattices'), 
                batch.get('cg_bonds'), batch.get('n_keypoint'), 
                batch.get('cluster'), batch.get('keypoint'))
  
