import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class Act(nn.Module):
  def __init__(self, act, slope=0.05):
    super(Act, self).__init__()
    self.act = act
    self.slope = slope
    self.shift = torch.log(torch.tensor(2.0)).item()

  def forward(self, input):
    if self.act == "relu":
      return F.relu(input)
    elif self.act == "leaky_relu":
      return F.leaky_relu(input)
    elif self.act == "sp":
      return F.softplus(input, beta=1)
    elif self.act == "leaky_sp":
      return F.softplus(input, beta=1) - self.slope * F.relu(-input)
    elif self.act == "elu":
      return F.elu(input, alpha=1)
    elif self.act == "leaky_elu":
      return F.elu(input, alpha=1) - self.slope * F.relu(-input)
    elif self.act == "ssp":
      return F.softplus(input, beta=1) - self.shift
    elif self.act == "leaky_ssp":
      return (
          F.softplus(input, beta=1) - 
          self.slope * F.relu(-input) - 
          self.shift
      )
    elif self.act == "tanh":
      return torch.tanh(input)
    elif self.act == "leaky_tanh":
      return torch.tanh(input) + self.slope * input
    elif self.act == "swish":
      return torch.sigmoid(input) * input
    else:
      raise RuntimeError(f"Undefined activation called {self.act}")

def reducer(x, reducer='max'):
  if reducer == 'max':
    return torch.max(x, dim=0, keepdim=True)[0]
  elif reducer == 'min':
    return torch.min(x, dim=0, keepdim=True)[0]
  elif reducer == 'sum':
    return torch.sum(x, dim=0, keepdim=True)
  elif reducer == 'mean':
    return torch.mean(x, dim=0, keepdim=True)
  else:
    raise NotImplementedError

def build_mlp(in_dim, units, layers, out_dim, 
              act='relu', layer_norm=False, act_final=False):
  mods = [nn.Linear(in_dim, units), Act(act)]
  for i in range(layers-1):
    mods += [nn.Linear(units, units), Act(act)]
  mods += [nn.Linear(units, out_dim)]
  if act_final:
    mods += [Act(act)]
  if layer_norm:
    mods += [nn.LayerNorm(out_dim)]
  return nn.Sequential(*mods)

class EdgeBlock(nn.Module):
  
  def __init__(self,
               net_kwargs):
    super(EdgeBlock, self).__init__()
    self._edge_mlp = build_mlp(**net_kwargs)
    
  def forward(self, graph):
    collected_edges = torch.cat([graph.edges, 
                                 graph.nodes[graph.receivers], 
                                 graph.nodes[graph.senders]], dim=1)
    updated_edges = self._edge_mlp(collected_edges)
    
    return graph.replace(edges=updated_edges)

class NodeBlock(nn.Module):
  
  def __init__(self,
               net_kwargs,
               reducer='sum'):
    """
    use node features and receiver messages. 
    """
    super(NodeBlock, self).__init__()
    self._node_mlp = build_mlp(**net_kwargs)
    self._reducer = reducer
    
  def forward(self, graph):
    receiver_agg = scatter(graph.edges, graph.receivers, dim=0, 
                           dim_size=graph.nodes.shape[0], reduce=self._reducer)
    nodes_to_collect = [graph.nodes, receiver_agg]
    collected_nodes = torch.cat(nodes_to_collect, dim=1)
    updated_nodes = self._node_mlp(collected_nodes)
    return graph.replace(nodes=updated_nodes)

class GraphIndependent(nn.Module):
  
  def __init__(self, 
               edge_net_kwargs=None,
               node_net_kwargs=None):
    super(GraphIndependent, self).__init__()
    self._edge_net_kwargs = edge_net_kwargs
    self._node_net_kwargs = node_net_kwargs
    if self._edge_net_kwargs:
      self._edge_mlp = build_mlp(**self._edge_net_kwargs)
    if self._node_net_kwargs:
      self._node_mlp = build_mlp(**self._node_net_kwargs)
  
  def forward(self, graph):
    
    if self._edge_net_kwargs:
      updated_edges = self._edge_mlp(graph.edges)
    else:
      updated_edges = graph.edges
    if self._node_net_kwargs:
      updated_nodes = self._node_mlp(graph.nodes)
    else:
      updated_nodes = graph.nodes
      
    return graph.replace(edges=updated_edges, nodes=updated_nodes)
    
class InteractionLayer(nn.Module):
  
  def __init__(self,
               edge_net_kwargs,
               node_net_kwargs,
               reducer='sum'):
    # global is concat rather than using a global block
    super(InteractionLayer, self).__init__()
    self._edge_net_kwargs = edge_net_kwargs
    self._node_net_kwargs = node_net_kwargs
    self._reducer = reducer
    self._edge_block = EdgeBlock(self._edge_net_kwargs)
    self._node_block = NodeBlock(self._node_net_kwargs, self._reducer)

  def forward(self, graph):
    graph = self._edge_block(graph)
    graph = self._node_block(graph)
    return graph
  
class GraphPooling(nn.Module):
  
  def __init__(self, 
               use_nodes=True, 
               use_edges=False, 
               use_globals=False,
               reducers=['max']):
    super(GraphPooling, self).__init__()
    self._use_nodes = use_nodes
    self._use_edges = use_edges
    self._use_globals = use_globals
    self._reducers = reducers
  
  def forward(self, nodes, n_node):
    nodes_per_graph_list = torch.split(nodes, list(n_node), dim=0)
    pooled_nodes_list = []
    for nodes_graph_i in nodes_per_graph_list:
      pooled_nodes_i = []
      for _reducer in self._reducers:
        pooled_nodes_i.append(reducer(nodes_graph_i, _reducer)) 
      pooled_nodes_i = torch.cat(pooled_nodes_i, dim=1)  # 1 x len(self._reducer)*dim
      pooled_nodes_list.append(pooled_nodes_i)
    pooled_nodes = torch.cat(pooled_nodes_list, dim=0)
    return pooled_nodes
class EncodeProcess(nn.Module):
  
  def __init__(self, node_dim, edge_dim, units, layers, latent_dim, mp_steps):
    super().__init__()
    self._units = units
    self._layers = layers
    self._latent_dim = latent_dim
    self._mp_steps = mp_steps
    self._net_kwargs = {
        'units': units,
        'layers': layers,
        'out_dim': latent_dim,
        'layer_norm': True
    }
    self._encode = GraphIndependent(
        edge_net_kwargs=dict(in_dim=edge_dim, **self._net_kwargs), 
        node_net_kwargs=dict(in_dim=node_dim, **self._net_kwargs))
    self._interaction = nn.ModuleList([
        InteractionLayer(
            edge_net_kwargs=dict(in_dim=latent_dim * 3, **self._net_kwargs), 
            node_net_kwargs=dict(in_dim=latent_dim * 2, **self._net_kwargs))
        for _ in range(self._mp_steps)
    ])

  def forward(self, input_graph):
    
    if input_graph.globals is not None:
      input_graph = input_graph.replace(
          nodes=torch.cat([input_graph.nodes, torch.repeat_interleave(
                              input_graph.globals, input_graph.n_node, dim=0)], dim=1), 
          globals=None)
    latent_graph = self._encode(input_graph)
    
    # Process
    for inter_layer in self._interaction:
      _latent_graph = inter_layer(latent_graph)
      # residual connection
      _latent_graph = _latent_graph.replace(
          nodes=_latent_graph.nodes+latent_graph.nodes,
          edges=_latent_graph.edges+latent_graph.edges)
      latent_graph = _latent_graph
    
    return latent_graph
    
class EncodeProcessDecode(EncodeProcess):
  
  def __init__(self, node_dim, edge_dim, units, layers, latent_dim, mp_steps, out_dim):
    super().__init__(node_dim, edge_dim, units, layers, latent_dim, mp_steps)
    self._out_net_kwargs = {
        'in_dim': latent_dim,
        'units': units,
        'layers': layers,
        'out_dim': out_dim
    }
    self._decode = build_mlp(**self._out_net_kwargs)
    
  def forward(self, input_graph, return_latent=False):
    latent_graph = super().forward(input_graph)
    output = self._decode(latent_graph.nodes)
    if return_latent:
      return output, latent_graph
    else:
      return output