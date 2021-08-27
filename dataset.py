import networkx as nx 
from bfs import BFS
from bellman_ford import BellmanFord
import numpy as np
from torch_geometric.utils import from_networkx
import torch 

bfs = BFS()
bellmanford = BellmanFord()


def get_trainloader_bellmanford(graph_type):  
  num_train = 100
  num_nodes = 20
  targets = []
  graphs = []
  predecessors = []
  for i in range(num_train):
    graph = GraphFactory.get_graph(num_nodes, graph_type)
    n = len(graph.nodes)
    root = np.random.randint(0, n)
    states, p, l = bellmanford.run(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.tensor(states[0])
    
    graphs.append(graph)
    targets.append(torch.tensor(states[1:]).float())
    predecessors.append(torch.tensor(p).int())
  return graphs, targets, predecessors

def get_valloader_bellmanford(graph_type):  
  num_val = 5
  num_nodes = 20
  targets = []
  graphs = []
  predecessors = []
  for i in range(num_val):
    graph = GraphFactory.get_graph(num_nodes, graph_type)
    n = len(graph.nodes)
    root = np.random.randint(0, n)
    states, p, l = bellmanford.run(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.tensor(states[0])
    
    graphs.append(graph)
    targets.append(torch.tensor(states[1:]).float())
    predecessors.append(torch.tensor(p).int())
  return graphs, targets, predecessors

def get_testloader_bellmanford(graph_type):  
  num_test = 5
  num_nodes = 100
  targets = []
  graphs = []
  predecessors = []
  for i in range(num_test):
    graph = GraphFactory.get_graph(num_nodes, graph_type)
    n = len(graph.nodes)
    root = np.random.randint(0, n)
    states, p, l = bellmanford.run(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.tensor(states[0])
    
    graphs.append(graph)
    targets.append(torch.tensor(states[1:]).float())
    predecessors.append(torch.tensor(p).int())
  return graphs, targets, predecessors



# BFS
def get_trainloader(graph_type):  
  num_train = 100
  num_nodes = 20
  targets = []
  graphs = []
  for i in range(num_train):
    graph = GraphFactory.get_graph(num_nodes, graph_type)
    n = len(graph.nodes)
    root = np.random.randint(0, n)
    states = bfs.run(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.zeros(n)
    
    graph.x[root] = 1
    graphs.append(graph)
    targets.append(torch.tensor(states).float())
  return graphs, targets

def get_valloader(graph_type):
  num_val = 5
  num_nodes = 20
  targets = []
  graphs = []
  for i in range(num_val):
    graph = GraphFactory.get_graph(num_nodes, graph_type)
    n = len(graph.nodes)
    root = np.random.randint(0, n)
    states = bfs.run(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.zeros(n)
    graph.x[root] = 1
    graphs.append(graph)
    targets.append(torch.tensor(states).float())
  return graphs, targets

def get_testloader(graph_type):
  num_test = 5
  num_nodes = 100
  targets = []
  graphs = []
  for i in range(num_test):
    graph = GraphFactory.get_graph(num_nodes, graph_type)
    n = len(graph.nodes)
    root = np.random.randint(0, n)

    states = bfs.run(graph,root)

    graph = from_networkx(graph)
    graph.x = torch.zeros(n)
    graph.x[root] = 1
    graphs.append(graph)
    targets.append(torch.tensor(states).float())
  return graphs, targets

class GraphFactory:
  @staticmethod
  def get_graph(n_nodes, type):
    '''
    Parameters
    ---------------------------
    n_nodes: int, number of nodes
    type: string, it has the possible values
         "ladder", "grid", "trees", "erdos_renyi", "barabasi_albert" 
         "4-community", "4-caveman"
    
    Results
    ---------------------------
    graph: nx.Graph
    '''
    graph = None 
    if type == 'ladder':
      graph = GraphFactory.get_ladder_graph(n_nodes)
    elif type == 'grid':
      graph = GraphFactory.get_grid_graph(n_nodes)
    elif type == 'trees':
      pass
    elif type == 'erdos_renyi':
      graph = GraphFactory.get_erdos_renyi_graph(n_nodes)
    elif type == 'barabasi_albert':
      graph = GraphFactory.get_barbasi_albert_graph(n_nodes)
    elif type == '4-community':
      pass
    elif type == '4-caveman':
      # TODO:
      pass
    GraphFactory._initialize_nodes_weights(graph)
    GraphFactory._initialize_edges_weights(graph)
    return graph

  @staticmethod
  def _initialize_edges_weights(graph):
    n_edges = len(graph.edges)
    weights = np.random.uniform(low=0.2, high=1.0, size=n_edges)
    edge_to_weight = {edge: weights[i] for i, edge in enumerate(graph.edges())}
    nx.set_edge_attributes(graph, edge_to_weight, 'weight')
  
  @staticmethod
  def _initialize_nodes_weights(graph):
    n_nodes = len(graph.nodes)
    weights = np.random.uniform(low=0.2, high=1.0, size=n_nodes)
    node_to_weight = {node: weights[i] for i, node in enumerate(graph.nodes())}
    nx.set_node_attributes(graph, node_to_weight, 'nodes')

  @staticmethod
  def get_ladder_graph(n_nodes):
    graph = nx.ladder_graph(n_nodes)
    return graph

  @staticmethod
  def get_grid_graph(n_nodes):
    n_nodes_row = int ( (n_nodes) ** 0.5) + 1
    n_nodes_col = n_nodes_row 
    graph = nx.grid_2d_graph(n_nodes_row, n_nodes_col)
    return graph
  
  @staticmethod 
  def get_erdos_renyi_graph(n_nodes):
    p = min(np.log2(n_nodes)  / n_nodes, 0.5)
    graph = nx.erdos_renyi_graph(n_nodes, p)
    return graph
  
  @staticmethod
  def get_barbasi_albert_graph(n_nodes):
    n_attached_nodes = 4
    graph = nx.barabasi_albert_graph(n_nodes, n_attached_nodes)
    return graph
  
  @staticmethod
  def get_4_community(n_nodes):
    n_nodes = n_nodes // 4
    graphs = [GraphFactory.get_erdos_renyi_graph(n_nodes) for _ in range(4)]
    
    graph = graphs[0]
    for i in range(1,len(graphs)):
      len_graph = len(graph.nodes)
      len_next_graph = len(graphs[i].nodes)
      graph = nx.union(graph, graphs[i], rename=('G', 'H'))
      
      G_nodes = ['G'+str(i) for i in range(len_graph)]
      H_nodes = ['H'+str(i) for i in range(len_next_graph)]
      size = len_graph * len_next_graph
      number_of_edges = np.sum(np.random.uniform(size=size) <= .01)
      g_nodes_to_connect = np.random.choice(G_nodes, replace=True, size=number_of_edges)
      h_nodes_to_connect = np.random.choice(H_nodes, replace=True, size=number_of_edges)
      edges = list(zip(g_nodes_to_connect, h_nodes_to_connect))
      graph.add_edges_from(edges)
    return graph

if __name__ == '__main__':
  g,s,p = get_trainloader_bellmanford('erdos_renyi')
  print(g[0].x)
  print(s[0])
  g,s,p = get_valloader_bellmanford('erdos_renyi')
  g,s,p = get_testloader_bellmanford('erdos_renyi')
  
      

  


  
  
    