import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv, NNConv,GatedGraphConv,MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU


class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(Encoder, self).__init__()
    self.w = nn.Linear(input_dim + hidden_dim, hidden_dim)
    self.relu=nn.ReLU()
  def forward(self, x, h):
    z = self.relu(self.w(torch.cat([x, h], dim=1)))# N, hidden_dim
    return z

# ==================== Gat Processor ================ #
class GATP(nn.Module):
  def __init__(self, hidden_dim, n_layers):
    super(GATP, self).__init__()
    self.n_layers = n_layers
    self.relu = nn.ReLU()
    self.convs = nn.ModuleList(
        [GATConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(n_layers)]
    )

    self.bs = nn.ModuleList(
        [nn.BatchNorm1d(hidden_dim) for _ in range(n_layers-1)]
    )

  def forward(self, z, edge_index):
    h = z
    for conv, bs in zip(self.convs[:-1], self.bs):
      h = self.relu(conv(h,edge_index))
    h = self.convs[-1](h, edge_index)
    return h

# ==================== MPNN Processor ================ #
class MPNNP(nn.Module):
  def __init__(self, hidden_dim, n_layers, aggr):
    super(MPNNP, self).__init__()
    self.n_layers = n_layers
    self.relu = nn.ReLU()
    self.convs = GatedGraphConv(out_channels=hidden_dim, aggr=aggr, num_layers=n_layers)

  def forward(self, z, edge_index):
    h = self.convs(z, edge_index)
    return h

# ====================Edge Conv ======================= #

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

class EdgeP(nn.Module):
  def __init__(self, hidden_dim, n_layers, aggr):
    super(EdgeP, self).__init__()
    self.n_layers = n_layers
    self.relu = nn.ReLU()
    self.convs = nn.ModuleList(
      [ EdgeConv(hidden_dim,hidden_dim) for _ in range(n_layers)])

  def forward(self, z, edge_index):
    h = z
    for conv in self.convs[:-1]:
      h = self.relu(conv(h,edge_index))
    h = self.convs[-1](h, edge_index)
    return h

# ==================== Decoder ======================= #
class Decoder(nn.Module):
  def __init__(self, hidden_dim):
    super(Decoder, self).__init__()
    self.w = nn.Linear(2*hidden_dim, hidden_dim)
    self.w1 = nn.Linear(hidden_dim, hidden_dim)
    self.bs = nn.BatchNorm1d(hidden_dim)
    self.relu = nn.ReLU()
    self.head = nn.Linear(hidden_dim, 1)
    self.sig = torch.nn.Sigmoid()
  def forward(self, h, z):
    out = self.w(torch.cat([h, z], dim=1)) # N, hidden_dim
    out = self.relu(out)
    out = self.relu(self.w1(out))
    y = self.head(out)
    return self.sig(y)

# ================== Termination ====================== #
class Termination(nn.Module):
  def __init__(self, hidden_dim):
    super(Termination, self).__init__()
    self.w = nn.Linear(hidden_dim, 1)
    self.sig = torch.nn.Sigmoid()
  def forward(self, h):
    # inputs: N, input_dim ?? 
    h_bar = torch.mean(h, dim=1)
    t = self.sig(self.w(h_bar)) # N, hidden_dim
    return t


class TemplateModel(nn.Module):
  def __init__(self, processor_type, input_dim, hidden_dim, n_layers, aggr, device):
    super(TemplateModel, self).__init__()
    self.encoder = Encoder(input_dim, hidden_dim)
    self.decoder = Decoder(hidden_dim)
    self.device = device
    self.hidden_dim = hidden_dim

    if processor_type == 'mpnn':
      self.processor = MPNNP(hidden_dim, n_layers, aggr)
    elif processor_type == 'gat':
      self.processor = GATP(hidden_dim, n_layers)
    elif processor_type == 'edge':
      self.processor = EdgeP(hidden_dim, n_layers, aggr)
    
    self.termination = Termination(hidden_dim)
  
  def forward(self, x, h, edge_index):
    z = self.encoder(x,h)
    h = self.processor(z, edge_index)
    out = self.decoder(h,z)
    t = self.termination(h)
    return out, t, h

def get_model(processor_type, input_dim, hidden_dim, n_layers, aggr, device):
  return TemplateModel(
    processor_type,
    input_dim,
    hidden_dim,
    n_layers,
    aggr,
    device
  ).to(device)