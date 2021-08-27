import torch
import numpy as np
import torch.nn.functional as F

def train_model_bellman(model, optimizer, schedular, criteria, train_graphs, train_states, val_graphs, val_states, teacher_ratio):
  t_losses = []
  device = model.device
  hidden_dim = model.hidden_dim
  model.train()
  for graph, states in zip(train_graphs,train_states):
    x, edge_index, edge_weight = graph.x.to(device), graph.edge_index.to(device), graph.weight.to(device)
    states = states.to(device)
    N = x.shape[0]
    h = torch.zeros(N, hidden_dim).to(device)
    num_nodes = len(x)
    x=x.unsqueeze(dim=1)
    losses = []
    for i in range(num_nodes):
      optimizer.zero_grad()
      out,t, h = model(x.float(), h, edge_index, edge_weight.float())
      stop = 0
      if len(states) - 1 == i:
        stop = 1
      #teacher
      if np.random.rand() < teacher_ratio:
        x = states[i].unsqueeze(dim=1)
      else:
        x = out
      target= states[i]
      out = out.squeeze(1)
      reg_loss = F.mse_loss(out,target)
      termination_target = torch.tensor(stop).to(device).unsqueeze(0)
      
      termination_loss =  F.binary_cross_entropy(t,termination_target.float())
      
      loss = reg_loss + termination_loss
  
      loss.backward()
      
      optimizer.step()
      schedular.step()
      h = h.detach()
      x = x.detach()
      losses.append(loss.item())
      if stop ==1:
        break
    mean_loss = np.mean(losses)
    t_losses.append(mean_loss)
  a = val_bellman(model, val_graphs, val_states)
  return np.mean(t_losses), a
      

def val_bellman(model, gs, ts):
  all_losses = []
  device = model.device
  hidden_dim = model.hidden_dim
  model.eval()
  with torch.no_grad():
    for graph, states in zip(gs,ts):
      x, edge_index, edge_weight = graph.x.to(device), graph.edge_index.to(device), graph.weight.to(device)
      states = states.to(device)
      N = x.shape[0]
      h = torch.zeros(N, hidden_dim).to(device)
      v = len(x)
      x=x.unsqueeze(dim=1)
      losses = []
      for i in range(v):
        out,t, h = model(x.float(), h, edge_index, edge_weight.float())
        stop = 0
        if len(states)-1 == i:
          stop = 1
        x = out
        target= states[i]
        out = out.squeeze(1)
        loss = F.mse_loss(out, target)
        losses.append(loss.item())
        if stop ==1:
          break
      loss = np.mean(losses)
      all_losses.append(loss)
  average_loss=np.mean(all_losses)
  return average_loss

# BFS
def train_model(model, optimizer, schedular, criteria, train_graphs, train_states, val_graphs, val_states, teacher_ratio):
  t_losses = []
  device = model.device
  hidden_dim = model.hidden_dim
  model.train()
  for graph, states in zip(train_graphs,train_states):
    x, edge_index = graph.x.to(device), graph.edge_index.to(device)
    states = states.to(device)
    N = x.shape[0]
    h = torch.zeros(N, hidden_dim).to(device)
    num_nodes = len(x)
    x=x.unsqueeze(dim=1)
    losses = []
    for i in range(num_nodes):
      optimizer.zero_grad()
      out,t, h = model(x, h, edge_index)
      stop = 0
      if len(states) - 1 == i:
        stop = 1
      #teacher
      if np.random.rand() < teacher_ratio:
        x = states[i].unsqueeze(dim=1)
      else:
        x = (out > .5).to(int)
      target= states[i]
      out = out.squeeze(1)
      classification_loss = criteria(out,target)
      termination_target = torch.tensor(stop).to(device).unsqueeze(0)
      
      termination_loss = criteria(t,termination_target.float())
      
      loss = classification_loss + termination_loss
  
      loss.backward()
      
      optimizer.step()
      schedular.step()
      h = h.detach()
      x = x.detach()
      losses.append(loss.item())
      if stop ==1:
        break
    mean_loss = np.mean(losses)
    t_losses.append(mean_loss)
  a,l = val(model, val_graphs, val_states)
  return np.mean(t_losses), a, l
      

def val(model, gs, ts):
  all_accu = []
  last = []
  device = model.device
  hidden_dim = model.hidden_dim
  model.eval()
  with torch.no_grad():
    for graph, states in zip(gs,ts):
      x, edge_index = graph.x.to(device), graph.edge_index.to(device)
      states = states.to(device)
      N = x.shape[0]
      h = torch.zeros(N, hidden_dim).to(device)
      v = len(x)
      x=x.unsqueeze(dim=1)
      accus = []
      for i in range(v):
        out,t, h = model(x, h, edge_index)
        stop = 0
        if len(states)-1 == i:
          stop = 1
        x = ( out > .5).to(int)
        target= states[i]
        out =( out > .5).to(int)
        out = out.squeeze(1)
        accu = torch.sum(out == target) / len(out)
        accus.append(accu.item())
        if stop ==1:
          break
      accu = np.mean(accus)
      all_accu.append(accu)
      last.append(accus[-1])
  average_accu=np.mean(all_accu)
  last_accu = np.mean(last)
  return average_accu, last_accu
      

