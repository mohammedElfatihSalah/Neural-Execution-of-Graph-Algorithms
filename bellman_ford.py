import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from collections import deque
from heapq import heappush, heappop
#from dataset import GraphFactory
from time import time

class BellmanFord:
    def run(self, graph, root=0):
       
        x = self.initialize_x(graph, root)
        p = self.initialize_p(graph, root)
        history = []
        ps = []
        longest_path = -1
        E = nx.to_numpy_matrix(graph)
        E=np.array(E)

        not_terminate=True
        history = []
        n_nodes = len(E)
        nodes   = list(range(n_nodes)) 
        dist = self.initialize_x(graph, root)
        predecessor = self.initialize_p(graph,root)
        history.append(dist.copy())
        while not_terminate:
          freezed_dist = dist.copy()
          for node in nodes:
            new_dist = freezed_dist[node]
            neighbours = np.where(E[node] > 0)[0]
            for n in neighbours:
              if new_dist > freezed_dist[n] + E[n, node]:
                new_dist = freezed_dist[n] + E[n, node] 
            dist[node] = new_dist
          if (dist == history[-1]).all():
            not_terminate=False
          else:
            history.append(dist.copy())
          
          for node in nodes:
            node_dist = dist[node]
            neighbours = np.where(E[node] > 0)[0]
            p = node
            for n in neighbours:
              if node_dist > dist[n] + E[n, node]:
                node_dist = dist[n] + E[n, node] 
                p=n
                predecessor[node] = p
          ps.append(predecessor.copy())
            
              
        return history, ps,  longest_path


    def initialize_p(self, graph, root=0):
      # Getting the adjacency matrix
      E = nx.to_numpy_matrix(graph)
      E=np.array(E)
      nb_nodes = graph.number_of_nodes()
      #calculating the longest shortest path
      longest = self.get_longest_shortest_path(graph, root)
      x = np.full(nb_nodes, 0)
      x[root] = root
      for node in range(nb_nodes):
        if node == root:
          continue
        neighbours = np.where(E[node] > 0)[0]
        first_neighbour = None
        if len(neighbours) > 0:
          first_neighbour = neighbours[0]
        else:
          first_neighbour = nb_nodes
        x[node] = first_neighbour
      return x



    def initialize_x(self, graph, root=0):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
        The graph on which the algorithm should be run
        root: index of the node that should be used as a root for the DFS
        Returns:
        --------
        Initialized numpy representation of the graph, as used by our DFS implementation
        '''

        nb_nodes = graph.number_of_nodes()
        longest = self.get_longest_shortest_path(graph, root)
        x = np.full(nb_nodes, longest+1)
        x[root] = 0
        return x

    def get_longest_shortest_path(self, graph, root):
      longest_path = -1
      E = nx.to_numpy_matrix(graph)
      E=np.array(E)
      
      queue = []
      queue.append((0, root))
      memory = set()
      while len(queue) > 0 and len(memory) < len(graph.nodes):
        dist, cur = heappop(queue)
        memory.add(cur)
        neighbours = np.where(E[cur] > 0)[0]
        for n in neighbours:
          if n not in memory:
            n_dist = dist + E[cur, n]
            heappush(queue, (n_dist, n))
            if n_dist > longest_path:
              longest_path = n_dist
      return longest_path
  

if __name__ == '__main__':
  start_time = time()
  graph = GraphFactory.get_graph(80 ,'erdos_renyi')
  bellmanford = BellmanFord()
  longest = bellmanford.get_longest_shortest_path(graph, 0)
  end_time = time()

  interval = end_time - start_time
  print(f"The shortest longest path {longest:.2f} generated in {interval:.2f} secnods".format(longest))
  print('Initializing x ... ')
  x = bellmanford.initialize_x(graph)
  print(x)
  
  print("Initializing p....")
  p = bellmanford.initialize_p(graph)
  print(p)

  print("Running....")
  hist, ps, l= bellmanford.run(graph, 0)
  for arr in hist:
    print(arr)
  #print(hist)

  print("predescor")
  for arr in ps:
    print(arr)
  print()
  print("Adjacency")
  E = nx.to_numpy_matrix(graph)
  E=np.array(E)
  for arr in E:
    print(arr)


