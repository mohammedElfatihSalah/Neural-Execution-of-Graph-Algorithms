from config import args
from dataset import get_testloader
from utils import load_model
from models import get_model
from utils import seed_everything
import torch 
import numpy as np

def test(model, test_graphs, test_states):
  t_accus = []
  device = model.device
  hidden_dim = model.hidden_dim
  model.eval()
  for graph_no, (graph, states) in enumerate(zip(test_graphs,test_states)):
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
      x = (out > .5).to(int)
      target= states[i]
      out =( out > .5).to(int)
      out = out.squeeze(1)
      accu = torch.sum(out == target) / len(out)
      accus.append(accu.item())
      if stop ==1:
        break
    a = np.mean(accus)
    l = accus[-1]
    print(f'Graph {graph_no} | Average Accuracy {a:.2f} | Last Accuracy {l:.2f}')
    t_accus.append(a)

  all_a = np.mean(t_accus)
  print(f'Average Accuracy Across All Graphs {all_a:.2f}')
  return np.mean(t_accus), t_accus[-1]
      

if __name__ == '__main__':
  # run tests
  seed_everything(123)
  model_path = './checkpoints/best_model.pt'
  test_graphs, test_states = get_testloader(args['graph_type'])
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
    args['input_dim'], 
    args['hidden_dim'], 
    args['n_layers'], 
    args['aggr'], device)
  model = load_model(model,model_path)
  
  test(model, test_graphs, test_states)

  
