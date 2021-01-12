import os
import pandas as pd
from scipy.io import loadmat
from scipy.io import loadmat
# from torch.utils.data import Dataset, DataLoader
import re
import yaml


# %%


class ActionData:

    def __init__(self, data_path, dim, nature):

        self.path = data_path
        self.dim = dim
        self.nature = nature

        self.data = self.load()

    def load(self):
        files1 = os.listdir(self.path)
        df = pd.DataFrame([])

        for j in files1:
            print(j)
            D = loadmat(self.path + j)

            if self.nature == 'skeleton':
                A1 = D['d_skel']
                A2 = A1.reshape((A1.shape[0] * A1.shape[1], A1.shape[2]))
                A = pd.DataFrame(A2).T
            elif self.nature == 'IMU':
                A = pd.DataFrame(D['d_iner'])

            k = list(map(int, re.findall(r'\d+', j)))

            ac = k[0]
            sub = k[1]
            exp = k[2]
            A.insert(self.dim, "subject", sub)
            A.insert(self.dim+1, "experience", exp)
            A.insert(self.dim+2, "action", ac)

            df = df.append(A)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index], self.data.iloc[index, -1]




