import os
import pandas as pd
import yaml
import pickle
import torch
import numpy as np

# %%


class FeatureData:

    def __init__(self, paths):

        self.data, self.labels = self.load(paths)



    def load(self,paths):

        # pickle_in = open("save\\x_train", "rb")
        # df = pickle.load(pickle_in)

        with open(paths[0], "rb") as f:
            data =torch.from_numpy(pickle.load(f))

        with open(paths[1], "rb") as f:
            lab = torch.from_numpy(pickle.load(f))

        return data, lab

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index], self.labels[index]