import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from collections import deque
from heapq import heappush, heappop
from dataset import GraphFactory
from time import time

class BellmanFord:
    def run(self, graph, root=0):
        '''
        Parameters
        ----------
        graph: NetworkX Graph instance
        The graph on which the algorithm should be run
        root: index of the node that should be used as root for the DFS
        Returns:
        --------
        The history of x (states) when executing the DFS algorithm, and the DFS
        output
        '''
        x = self.initialize_x(graph, root)
        p = self.initialize_p(graph, root)
        history = []
        predecessor = []
        longest_path = -1
        E = nx.to_numpy_matrix(graph)
        E=np.array(E)
       
        queue = []
        queue.append((0, root))
        memory = set()
        while len(queue) > 0:
          dist, cur = heappop(queue)
          memory.add(cur)
          neighbours = np.where(E[cur] > 0)[0]
          for n in neighbours:
            if n not in memory:
              n_dist = dist + E[cur, n]
              heappush(queue, (n_dist, n))
              if n_dist > longest_path:
                longest_path = n_dist
              x[n] = min(x[n], n_dist)
              p[n] = cur
          history.append(x.copy())
          predecessor.append(p.copy())
        
        
        return longest_path, history, predecessor

    def initialize_p(self, graph, root=0):
      nb_nodes = graph.number_of_nodes()
      longest = self.get_longest_shortest_path(graph, root)
      x = np.full(nb_nodes, 0)
      x[root] = root
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
  graph = GraphFactory.get_graph(5 ,'erdos_renyi')
  bellmanford = BellmanFord()
  longest = bellmanford.get_longest_shortest_path(graph, 0)
  end_time = time()

  interval = end_time - start_time
  print(f"The shortest longest path {longest:.2f} generated in {interval:.2f} secnods".format(longest))
  print('Initializing x ... ')
  x = bellmanford.initialize_x(graph)
  print(x)
  print("Running..")
  l, hist,p = bellmanford.run(graph)
  E = nx.to_numpy_matrix(graph)

  for arr in hist:
    print(arr)
  print()

  for arr in p:
    print(arr)

  print()
  print("E")
  print(E)



