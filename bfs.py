import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from collections import deque
#from dataset import GraphFactory
from utils import seed_everything
class BFS:
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

        E = nx.to_numpy_matrix(graph)
        E=np.array(E)
        
        x = self.initialize_x(graph, root)
        history = [x.copy()]

        queue = deque()
        queue.append(root)
        memory = set()
        terminate=False
        while len(queue) > 0 and np.sum(x) < len(x):
          second_queue = deque()
          while len(queue) > 0 and np.sum(x) < len(x):
            cur = queue.popleft()
            #print("cur",cur)
            memory.add(cur)
            neighbours = np.where(E[cur] > 0)[0]
            #print(E[cur])
            for n in neighbours:
              #print("n",n)
              if n not in memory:
                #print("added")
                second_queue.append(n)
                x[n] = 1
          if (x == history[-1]).all():
            terminate = True
            break
          history.append(x.copy())
          queue = second_queue
          if terminate:
            break
        return np.asarray(history)

    

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
        x = np.zeros((nb_nodes))
        x[root] = 1

        return x

if __name__ == '__main__':
  # seed=3
  # seed_everything(seed)
  graph = GraphFactory.get_graph(7, 'erdos_renyi')
  E = nx.to_numpy_matrix(graph)
  E=np.array(E)

  bfs = BFS()
  hist = bfs.run(graph)
  for arr in hist:
    print(arr)
  
  print(E)
