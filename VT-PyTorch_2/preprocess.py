#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import pandas as pd

from six.moves import xrange  # pylint: disable=redefined-builtin


if __name__ == '__main__':
    root_dir = '/data/yyl/source/VT-PyTorch'
    data_dir = os.path.join(root_dir, 'dev_data/val_ori')
    dest_dir = os.path.join(root_dir, 'dev_data/val_rs')

    # create the subfolders for each class
    # for i in range(1000):
    #     os.mkdir(os.path.join(dest_dir, str(i+1)))

    # move all files to their corresponding subfolders
    labels = pd.read_csv('./dev_data/val_rs.csv').to_numpy()
    for fname, label in labels:
        ori_dir = os.path.join(data_dir, fname)
        new_dir = os.path.join(dest_dir, str(label))
        new_dir = os.path.join(new_dir, fname)
        shutil.copy(ori_dir, new_dir)