import math
import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl

from tqdm import tqdm
from torch_scatter import scatter

import graphwm.model.networks as nets
import graphwm.model.utils as utils
from graphwm.model.graphs import GraphsTuple

class GraphSim(pl.LightningModule):
  """
  GNN simulator base class
  """
  def __init__(self, *args, **kwargs) -> None:
    super().__init__()
    self.save_hyperparameters()
    self.latent_dim = self.hparams.dynamics_gn_hparams['latent_dim']
    self.use_coarse_graining = (self.hparams.cg_level > 1) or self.hparams.use_keypoint_for_cg

    # particle masses.
    if self.hparams.use_atom_mass:
      self.particle_masses = torch.FloatTensor(utils.ATOM_MASSES)
    else:
      # all particles have mass = 1.
      self.particle_masses = torch.FloatTensor(utils.PSEUDO_MASSES)
    
    # graph networks.
    if self.hparams.embedding_gn_hparams:
      self.embedding_gn = nets.EncodeProcessDecode(
            node_dim=self.hparams.type_emb_size+int(self.hparams.use_weights),
            edge_dim=self.hparams.type_emb_size,  
            out_dim=self.latent_dim, 
            **self.hparams.embedding_gn_hparams)
    
    # node/edge embeddings.
    self.type_embedding = nn.Embedding(self.hparams.num_ptypes, self.hparams.type_emb_size)
    nn.init.xavier_uniform_(self.type_embedding.weight)
    
    # types of fine level bonds. set <num_btypes==0> to disable bond type information.
    if self.hparams.num_btypes > 0:
      self.bond_type_embedding = nn.Embedding(self.hparams.num_btypes, self.hparams.type_emb_size)
      nn.init.xavier_uniform_(self.bond_type_embedding.weight)
    
    self.edge_type_embedding = nn.Embedding(2, self.hparams.type_emb_size)  # 2 types: cg-bonds vs radius edge.   
    nn.init.xavier_uniform_(self.edge_type_embedding.weight)
    
    # property head if applicable.
    if self.hparams.property_net_hparams:
      self.graph_pool = nets.GraphPooling(reducers=['mean'])
      self.property_head = nets.build_mlp(in_dim=self.latent_dim, 
                                          **self.hparams.property_net_hparams)
    
    # noise scales.
    self.noise_sigmas = torch.linspace(math.log(self.hparams.noise_begin), 
                                 math.log(self.hparams.noise_end), 
                                 self.hparams.noise_level).exp()
    self.noise_sigmas.requires_grad = False
  
  def predict(self):
    raise NotImplementedError
  
  def forward(self):
    raise NotImplementedError
  
  def step(self, batch, batch_idx: int):
    """take corresponding entries from batch to use in <forward>."""
    return NotImplementedError
  
  def coarse_graining(self, pos_seq, next_pos, ptype_embeddings, weights, 
                      cluster=None, keypoint=None):
    """
    coarse grain <pos_seq>, <next_pos>, <ptype_embeddings>, and <weights> using the 
    fine -> coarse mapping given by <cluster>.
    for systems with PBC only use the **unwrapped** positions as inputs.
    """
    pos_seq = torch.cat([pos_seq, next_pos[:, None, :]], dim=1)  # process next_pos at the same time.
    if 'use_keypoint_for_cg' in self.hparams and self.hparams.use_keypoint_for_cg:
      cg_pos_seq = pos_seq[keypoint]
      weights = weights.view(-1, 1, 1)  
      cg_weights = scatter(weights.squeeze(), cluster, dim=0, reduce='sum').view(-1, 1)
    else:
      cg_pos_seq, cg_weights = utils.compute_com(pos_seq, weights, cluster)
    cg_pos_seq, cg_next_pos = cg_pos_seq[:, :-1], cg_pos_seq[:, -1]
    cg_ptype_embeddings = scatter(ptype_embeddings, cluster, dim=0, reduce='mean')
    return cg_pos_seq, cg_next_pos, cg_ptype_embeddings, cg_weights
  
  def noise_augment(self, pos_seq, next_pos, n_node):
    """
    augment <pos_seq> and <next_pos> with noise.
    """
    if self.hparams.noise_method == 'uncorrelated':
      sampled_noise_seq = utils.get_position_noise(pos_seq, self.hparams.noise_begin, use_rw=False)
    elif self.hparams.noise_method == 'random_walk':
      sampled_noise_seq = utils.get_position_noise(pos_seq, self.hparams.noise_begin, use_rw=True)
    elif self.hparams.noise_method == 'multi_scale':
      noise_level = torch.randint(0, self.hparams.noise_level, 
                                  (n_node.shape[0],), device=n_node.device)
      sampled_sigmas = self.noise_sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1, 1)
      sampled_noise_seq = utils.get_position_noise(pos_seq, sampled_sigmas, use_rw=False)
    elif self.hparams.noise_method == 'multi_scale_random_walk':
      noise_level = torch.randint(0, self.hparams.noise_level, 
                                  (n_node.shape[0],), device=n_node.device)
      sampled_sigmas = self.noise_sigmas[noise_level.repeat_interleave(n_node)].view(-1, 1, 1)
      sampled_noise_seq = utils.get_position_noise(pos_seq, sampled_sigmas, use_rw=True)
    else:
      return pos_seq, next_pos
    
    # add noise.
    pos_seq = pos_seq + sampled_noise_seq
    if self.hparams.noise_target == 'vel':
      next_pos = next_pos + sampled_noise_seq[:, -1]
    else:
      assert self.hparams.noise_target == 'pos', '<noise_target> should be either <vel> or <pos>'
      
    return pos_seq, next_pos
  
  def wrap_batch(self, pos_seq, lattices, n_node):
    """
    warp batched coordinates in the lattices.
    """
    lattices = lattices.repeat_interleave(n_node, dim=0)
    return utils.wrap_positions(pos_seq, lattices)
  
  def _embedding_preprocessor(self, ptypes, n_node, bonds=None, btypes=None, weights=None):
    """
    compute node embedding at the fine graph level. excute before coarse-graining.
    only applicable when the ground truth graph has bond information.
    set <btype> when bond type is available. 
    if <embedding_gn_hparams> is <None>, use <ptype_embedding> as node embeddings.
    """
    ptype_embeddings = self.type_embedding(ptypes)
    if not self.hparams.embedding_gn_hparams:
      return ptype_embeddings
    else:
      assert bonds is not None, 'an embedding network should only be used when the \
                                 ground truth graph contains bonds.'                        
      senders, receivers = bonds[:, 0], bonds[:, 1]
      edge_embeddings = ptype_embeddings[senders] + ptype_embeddings[receivers]
      
      if btypes is not None and self.hparams.num_btypes > 0:
        edge_embeddings += self.bond_type_embedding(btypes)
      
      if self.hparams.use_weights:
        ptype_embeddings = torch.cat([ptype_embeddings, weights.view(-1, 1)], dim=-1)
        
      graph_tuple = GraphsTuple(
          nodes=ptype_embeddings,
          edges=edge_embeddings,
          edge_type=None,
          coords=None,
          globals=None,
          n_node=n_node,
          senders=senders,
          receivers=receivers)
      return self.embedding_gn(graph_tuple)
  
  def _dynamics_preprocessor(self, pos_seq, ptype_embeddings, n_node,
                             bonds=None, lattices=None, globals=None, weights=None):
    """
    Construct input graph for the dynamics modules.
    Compute connectivities, displacements and distances with periodic boundary conditions 
    if <lattices> is not <None>.
    """
    # node features.
    vel_seq = self._time_diff(pos_seq)  # Finite-difference.
    flat_vel_seq = torch.reshape(vel_seq, [vel_seq.shape[0], math.prod(vel_seq.shape[1:])])
    
    if self.hparams.use_weights and weights is not None:
      node_features = torch.cat([flat_vel_seq, ptype_embeddings, weights], dim=-1)
    else:
      node_features = torch.cat([flat_vel_seq, ptype_embeddings], dim=-1)
    
    # wrap positions under periodic boundary condition into lattices.
    if lattices is not None:
      pos_seq = self.wrap_batch(pos_seq, lattices, n_node)
    most_recent_pos = pos_seq[:, -1]
    
    # obtain connectivity.
    if lattices is None:
      senders, receivers, displacements, distances, edge_types = utils.compute_connectivity(
          most_recent_pos, n_node, self.hparams.radius, bonds)
    else:
      senders, receivers, displacements, distances, edge_types = utils.compute_connectivity_pbc(
          most_recent_pos, lattices, n_node, self.hparams.radius, bonds)
    edge_embeddings = self.edge_type_embedding(edge_types)
    edge_features = torch.cat([displacements, distances, edge_embeddings], dim=-1)
        
    return GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        edge_type=edge_types,
        coords=None,
        globals=None,
        n_node=n_node,
        senders=senders,
        receivers=receivers)
  
  def _time_diff(self, input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]
  
  def predict_prop(self, latent_graph, pos, weights, n_node, prop='rgs'):
    """
    predict the the specified property if applicable.
    For certain properties, we can compute from the CG state and fit the
    residual using a neural network on top of the latent graph.
    """
    pooled_graph = self.graph_pool(latent_graph.nodes.detach(), 
                                   latent_graph.n_node.detach())
    pred = self.property_head(pooled_graph)
    if prop == 'rgs':
      pred = pred + utils.compute_weighted_rgs(pos.detach(), weights.detach(), n_node)
    return pred
  
  def simulate(self, batch, rollout_length, 
               deter=False, save_positions=False, save_frequency=1,
               disable_bar=False):
    """
    simulate CGMD from a data batch of length <rollout_length>.
    deter:          if <True>, predict deterministically.
    save_positions: if <True>, save the rollout bead positions.
    save_frequency: frameskip when recording the trajectory.
    disable_bar:    if <True>, disable the progress bar.
    """
    output = {}
    pos_seq, next_pos = batch['position'], batch['target']
    pos_seq = pos_seq[:, :self.hparams.seq_len]
    if self.particle_masses.device != pos_seq.device:
      self.particle_masses = self.particle_masses.to(pos_seq.device)
    
    n_node, ptypes = batch['n_particle'], batch['particle_types']
    bonds = batch['bonds']
    weights = self.particle_masses[ptypes]
    ptype_embeddings = self._embedding_preprocessor(ptypes, n_node, bonds, batch.get('bond_types'), weights)
   
    if self.use_coarse_graining:
      (pos_seq, next_pos, ptype_embeddings, weights) = self.coarse_graining(
          pos_seq, next_pos, ptype_embeddings, weights, 
          batch.get('cluster'), batch.get('keypoint'))
      n_node = batch.get('n_keypoint')
      bonds = batch.get('cg_bonds')
    output['cg_u_pos'] = pos_seq.detach().clone().cpu()
    output['cg_weights'] = weights
    
    all_positions = []
    all_props = []
    for t in tqdm(range(rollout_length), disable=disable_bar):
      next_pos_pred, prop_pred = self.predict(
          pos_seq, n_node, ptype_embeddings, bonds, weights, batch.get('lattices'), deterministic=deter)
      pos_seq = torch.cat([pos_seq[:, 1:], next_pos_pred.unsqueeze(1)], dim=1)
      if t % save_frequency == 0:
        all_positions.append(next_pos_pred.detach().cpu().unsqueeze(1))
      if prop_pred is not None:
        all_props.append(prop_pred.detach().cpu())

    if save_positions:
      output['rollout_u_pos'] = torch.cat(all_positions, dim=1)
    
    if prop_pred is not None:
      output['rollout_prop'] = torch.cat(all_props, dim=1)
    
    return output
  
  # pytorch-lightning function
  def configure_optimizers(self):
    opt = hydra.utils.instantiate(
        self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
    )
    if not self.hparams.optim.use_lr_scheduler:
      return [opt]
    scheduler = hydra.utils.instantiate(
        self.hparams.optim.lr_scheduler, optimizer=opt
    )
    
    lr_dict = {
        'scheduler': scheduler,  # The LR scheduler instance (required)
        'interval': self.hparams.optim.scheduler_interval,  # The unit of the scheduler's step size, could also be 'step'
        'frequency': 1,  # The frequency of the scheduler
        'monitor': 'val_loss',  # Metric for `ReduceLROnPlateau` to monitor
        'strict': True,  # Whether to crash the training if `monitor` is not found
        'name': None,  # Custom name for `LearningRateMonitor` to use
    }
    
    return [opt], [lr_dict]

  # pytorch-lightning function
  def training_step(self, batch, batch_idx: int) -> torch.Tensor:
    loss_dict = self.step(batch, batch_idx)
    self.log_dict(
        {'train_'+k: v for k, v in loss_dict.items()},
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )
    return loss_dict['loss']

  # pytorch-lightning function
  def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
    loss_dict = self.step(batch, batch_idx)
    self.log_dict(
        {'val_'+k: v for k, v in loss_dict.items()},
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    return loss_dict['loss']

  # pytorch-lightning function
  def test_step(self, batch, batch_idx: int) -> torch.Tensor:
    loss_dict = self.step(batch, batch_idx)
    self.log_dict(
          {'test_'+k: v for k, v in loss_dict.items()}
    )
    return loss_dict['loss']
