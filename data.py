import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os

class ShapeNetDataset(data.Dataset):

    def __init__(self, train=True, npoints=8192):
        self.train = train
        self.npoints = npoints

        if train:
            list_path = './data/train.list'
        else:
            list_path = './data/val.list'

        with open(list_path) as f:
            self.model_list = [line.strip().replace('/', '_') for line in f]

        self.indices = torch.randperm(len(self.model_list) * 50)
        self.len = len(self.indices)

    def __getitem__(self, index):
        model_id = self.model_list[self.indices[index] // 50]
        scan_id = self.indices[index] % 50

        def read_pcd(filename):
            try:
                pcd = o3d.io.read_point_cloud(filename)
                return torch.from_numpy(np.array(pcd.points)).float()
            except:
                return None

        if self.train:
            partial = read_pcd(f"./data/train/{model_id}_{scan_id}_denoised.pcd")
        else:
            partial = read_pcd(f"./data/val/{model_id}_{scan_id}_denoised.pcd")

        complete = read_pcd(f"./data/complete/{model_id}.pcd")

        if partial is None or complete is None:
            return None

        return model_id, ShapeNetDataset.resample_pcd(partial, 5000), ShapeNetDataset.resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len

    @staticmethod
    def resample_pcd(pcd, n):

        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
        return pcd[idx[:n]]
