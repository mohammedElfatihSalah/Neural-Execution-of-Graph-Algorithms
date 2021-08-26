import torch
import random
import numpy as np
import time
import os

def seed_everything(seed):
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)


def save_model(model, args):
  PATH = args['PATH']
  lr = args['lr']
  average_accuracy = args['average_accuracy']
  last_accuracy = args['last_accuracy']
  model_type = args['processor_type']
  current_time = time.time()
  current_time = str(current_time)
  current_time = current_time[:3]
  
  model_name = "{0}_average_accu_{1:.2f}_last_accuracy_{2:.2f}_lr_{3}.pt".format(
    model_type,
    average_accuracy,
    last_accuracy,
    lr
  )

  model_name = 'best_model.pt'

  if not os.path.isdir(PATH):
    os.mkdir(PATH)
  
  path_to_save = os.path.join(PATH,model_name)

  torch.save(model.state_dict(), path_to_save)




def load_model(model, path):
  model.load_state_dict(torch.load(path))
  return model
