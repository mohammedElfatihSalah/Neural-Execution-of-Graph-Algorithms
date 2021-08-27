args = {
  'PATH' : './checkpoints',
  'graph_type' :'grid', # ladder, erdos_renyi, grid
  'task': 'bellman' , # bfs, bellmanford
  'processor_type' : 'mpnn', # edge, gat, mpnn
  'aggr':'max',
  'teacher_ratio':0.0,
  'n_epochs':500,
  'input_dim':1,
  'hidden_dim':32,
  'n_layers':1,
  'lr': .0005}