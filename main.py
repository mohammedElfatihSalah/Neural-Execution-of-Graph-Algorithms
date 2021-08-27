import time
import argparse
from dataset import *
from models import get_model
import torch
import torch.optim as optim
import torch.nn as nn
from train import train_model,train_model_bellman
from utils import seed_everything, save_model
from config import args
import torch_sparse
import torch_scatter
def main(args):
  pass

def train_bfs(args):
    print('Seeding....')
    seed_everything(1234)
    print()
    print('Preparing data....')
    start_time = time.time()
    train_graphs, train_states = get_trainloader(args['graph_type'])
    val_graphs, val_states = get_valloader(args['graph_type'])
    test_graphs, test_states = get_testloader(args['graph_type'])
    end_time = time.time()
    print(f'Prepared in {end_time - start_time:.2f} seconds')
    print()
    print('Checking GPU ... ')
    if torch.cuda.is_available():
      print('cuda is available')
      device = torch.device("cuda")
    else:
      print('cuda is not available')
      device = torch.device("cpu")
    print()

    model = get_model(
      args['processor_type'], 
      args['task'],
      args['input_dim'], 
      args['hidden_dim'], 
      args['n_layers'], 
      args['aggr'], device)

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    cosine_schedualr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criteria = nn.BCELoss()

    best_model = None
    best_accu = -1
    PATH='./best_model.pt'
    patience = 0
    MAX_PATIENCE=15

  
    for i in range(args['n_epochs']):
      loss,a,v = train_model(model,optimizer, cosine_schedualr,
      criteria, train_graphs, train_states, val_graphs, val_states, args['teacher_ratio'])
      print(f"Epoch {i+1} | Train loss {loss:.2f} | Val accuracy {a*100:.2f}/{v*100:.2f} | Best {best_accu*100:.2f}")
      if best_accu < a:
        patience=0
        args['average_accuracy'] = a
        args['last_accuracy'] = v
        save_model(model, args)
        best_accu = a
      else:
        patience +=1
        if patience == MAX_PATIENCE:
          break

def train_bellman(args):
    print('Seeding....')
    seed_everything(1234)
    print()
    print('Preparing data....')
    start_time = time.time()
    train_graphs, train_states,_ = get_trainloader_bellmanford(args['graph_type'])
    val_graphs, val_states,_ = get_valloader_bellmanford(args['graph_type'])
    test_graphs, test_states,_ = get_testloader_bellmanford(args['graph_type'])
    end_time = time.time()
    print(f'Prepared in {end_time - start_time:.2f} seconds')
    print()
    print('Checking GPU ... ')
    if torch.cuda.is_available():
      print('cuda is available')
      device = torch.device("cuda")
    else:
      print('cuda is not available')
      device = torch.device("cpu")
    print()

    model = get_model(
      args['processor_type'], 
      args['task'],
      args['input_dim'], 
      args['hidden_dim'], 
      args['n_layers'], 
      args['aggr'], device)

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    cosine_schedualr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criteria = nn.BCELoss()

    best_model = None
    best_loss = 1e100
    PATH='./best_model.pt'
    patience = 0
    MAX_PATIENCE=10

    print("Start trainning...")
    for i in range(args['n_epochs']):
      loss,a = train_model_bellman(model,optimizer, cosine_schedualr,
      criteria, train_graphs, train_states, val_graphs, val_states, args['teacher_ratio'])
      print(f"Epoch {i+1} | Train loss {loss:.2f} | Val loss {a*100:.2f} | Best {best_loss*100:.2f}")
      if best_loss > a:
        patience=0
        args['average_accuracy'] = a
        args['last_accuracy'] = 0
        save_model(model, args)
        best_loss = a
      else:
        patience +=1
        if patience == MAX_PATIENCE:
          break

if __name__ == '__main__':
  if args['task'] == 'bfs':
    train_bfs(args)
  elif args['task'] == 'bellman':
    train_bellman(args)
    
  
    
    

