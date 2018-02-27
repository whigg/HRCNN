"""Split the dataset into train/dev/test"""


import random
import os
import sys
from toolbox.paths import data_dir
import shutil as st


# Variable considered
var = sys.argv[1]


# Define the data directories
train_dir = 'Train_' + var  # Contains data
test_dir = 'Test_' + var
dev_dir = 'Dev_' + var
train_dir = os.path.join(data_dir, train_dir)
dev_dir = os.path.join(data_dir, dev_dir)
test_dir = os.path.join(data_dir, test_dir)
assert os.path.isdir(train_dir), "Couldn't find the dataset at {}".format(train_dir)


# Get the filenames in train directory
filenames = os.listdir(train_dir)
filenames = [os.path.join(train_dir, f) for f in filenames if f.endswith('.nc')]

# Split the data into 80% train, 10% dev, 10% test
# Make sure to always shuffle with a fixed seed so that the split is reproducible
random.seed(230)
filenames.sort()
random.shuffle(filenames)
split = int(0.8 * len(filenames))
train_filenames = filenames[:split]
filenames = filenames[split:]
split = int(0.5 * len(filenames))
dev_filenames = filenames[:split]
test_filenames = filenames[split:]

print("Processing {} data, saving preprocessed data to {}".format('test', test_dir))
for filename in test_filenames:
    outfile = test_dir + '/' + filename.split('/')[-1]
    st.move(filename,outfile)
print("Processing {} data, saving preprocessed data to {}".format('dev', dev_dir))
for filename in dev_filenames:
    outfile = dev_dir + '/' + filename.split('/')[-1]
    st.move(filename,outfile)

print("Done building dataset")
