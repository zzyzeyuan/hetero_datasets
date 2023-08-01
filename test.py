# created by zzy
# test pyg loaded data

from torch_geometric.datasets import IMDB, DBLP
import torch

import os
import os.path as osp


import numpy as np


dataset = IMDB(root='dd/IMDB')
dataset_dblp = DBLP(root='dd/DBLP')
print(dataset[0])
# print(dataset_dblp[0])
