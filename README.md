# Graph-Execution-of-Graph-Algorithms
It's unofficial implementation of the paper [Neural Exexcution of Graph Algorithms](https://arxiv.org/abs/)

![alt text](https://github.com/mohammedElfatihSalah/Graph-Execution-of-Graph-Algorithms/blob/master/paper_abstract.png "Logo Title Text 1")

# Prerequisites
To install required dependencies run the script
```
sh torch_geometric_install.sh
```

# Configuration
You can change configuration in `config.py`. Models that are available are
- GAT as in the paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903).
- GatedGraphConv [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).

and tasks that models can learn are:
- Reachability (BFS).
- BellmanFord the shortest distance to source node.

# Train 
To train a model just run the script
```
python main.py
```

# Results
My results for graph category of `erdos_renyi` with 100 nodes after training on only 20 nodes for reachability task.
| Tables        | My Results           |
| ------------- |:-------------:| 
| GAT    | 99/100% | 
| MPNN-max      | 100/100%      | 

