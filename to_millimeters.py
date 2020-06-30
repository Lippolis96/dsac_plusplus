import os
from utils_global import posename as posename
import numpy as np
import pdb

print('are you sure you want to update the dataset from meters to millimeters')
print('if you are sure press c + Enter')
pdb.set_trace()


sequences = ['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06']
for sequence in sequences:
    s_path = os.path.join(os.path.join('data', sequence), 'poses')
    for i in range(1000):
        posefile = os.path.join(s_path, posename(i))
        pose = np.loadtxt(posefile)
        pose[:3,3] = pose[:3,3]*1000
        np.savetxt(posefile, pose)