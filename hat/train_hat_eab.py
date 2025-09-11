# flake8: noqa
import os.path as osp

import hat.archs
import hat.data
import hat.models
import hat.data.paired_image_dataset_edge
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
