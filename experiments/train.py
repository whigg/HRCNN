import argparse
from functools import partial
import json
from keras import optimizers
from pathlib import Path
import sys
sys.path.insert(0, "/data/thomas/bdelorme/DL_project/HR_CNN/toolbox")
from data import load_set
from models import get_model
from experiment import Experiment


parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=Path)
args = parser.parse_args()
param = json.load(args.param_file.open())

# Model
scale = param['scale']
build_model = partial(get_model(param['model']['name']), **param['model']['params'])
if 'optimizer' in param:
    optimizer = getattr(optimizers, param['optimizer']['name'].lower())
    optimizer = optimizer(**param['optimizer']['params'])
else:
    optimizer = 'adam'

# Data
load_set = partial(load_set, lr_sub_size=param['lr_sub_size'], lr_sub_stride=param['lr_sub_stride'])

# Training
expt = Experiment(scale=param['scale'], load_set=load_set, build_model=build_model, optimizer=optimizer, save_dir=param['save_dir'])
expt.train(train_set=param['train_set'], dev_set=param['dev_set'], epochs=param['epochs'], resume=True)

