"""Split the dataset into train/dev/test"""


import random
import os
import sys
from paths import data_dir
import shutil as st


# Variable considered
var = sys.argv[1]


# Define the data directories
train_dir = 'Train_' + var
test_dir = 'Test_' + var
dev_dir = 'Dev_' + var
apply_dir = 'Apply_' + var
res_apply_dir = 'Res_Apply_' + var
res_train_dir = 'Res_Train_' + var
train_dir = os.path.join(data_dir, train_dir)
dev_dir = os.path.join(data_dir, dev_dir)
test_dir = os.path.join(data_dir, test_dir)
apply_dir = os.path.join(data_dir, apply_dir)
res_apply_dir = os.path.join(data_dir, res_apply_dir)
res_train_dir = os.path.join(data_dir, res_train_dir)


# train directory
filenames = os.listdir(train_dir)
filenames = [os.path.join(train_dir, f) for f in filenames if f.endswith('.nc')]
random.seed(230)
filenames.sort()
random.shuffle(filenames)
split = int(0.1 * len(filenames))
res_filenames = filenames[split:]
for filename in res_filenames:
    outfile = res_train_dir + '/' + filename.split('/')[-1]
    st.move(filename,outfile)
# dev directory
filenames = os.listdir(dev_dir)
filenames = [os.path.join(dev_dir, f) for f in filenames if f.endswith('.nc')]
random.seed(230)
filenames.sort()
random.shuffle(filenames)
split = int(0.1 * len(filenames))
res_filenames = filenames[split:]
for filename in res_filenames:
    outfile = res_train_dir + '/' + filename.split('/')[-1]
    st.move(filename,outfile)
# test directory
filenames = os.listdir(test_dir)
filenames = [os.path.join(test_dir, f) for f in filenames if f.endswith('.nc')]
random.seed(230)
filenames.sort()
random.shuffle(filenames)
split = int(0.1 * len(filenames))
res_filenames = filenames[split:]
for filename in res_filenames:
    outfile = res_train_dir + '/' + filename.split('/')[-1]
    st.move(filename,outfile)
# apply directory
filenames = os.listdir(apply_dir)
filenames = [os.path.join(apply_dir, f) for f in filenames if f.endswith('.nc')]
random.seed(230)
filenames.sort()
random.shuffle(filenames)
split = int(0.1 * len(filenames))
res_filenames = filenames[split:]
for filename in res_filenames:
    outfile = res_apply_dir + '/' + filename.split('/')[-1]
    st.move(filename,outfile)

