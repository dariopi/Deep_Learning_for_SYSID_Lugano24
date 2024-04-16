# create custome DataSet
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat  # Add this import




class F16DS(Dataset):

    def __init__(self, pd_file, na, nb, ind_out):
        output_names = ['Acceleration1', 'Acceleration2', 'Acceleration3']
        self.pd_file = pd_file  # pandas dataset
        self.na = na
        self.nb = nb

        n = max(na, nb)
        self.regressor = np.zeros((len(pd_file) - n, (nb + 1) * 2 + na))

        for i in range(n, len(pd_file) - n):
            self.regressor[i - n, 0:nb + 1] = pd_file.loc[i:i - nb:-1, 'Force']
            self.regressor[i - n, nb + 1:nb + nb + 2] = pd_file.loc[i:i - nb:-1, 'Voltage']
            self.regressor[i - n, nb + nb + 2:nb + nb + 2 + na] = pd_file.loc[i - 1:i - na:-1, output_names[ind_out]]

        self.regressor = torch.from_numpy(self.regressor).float()

        self.out = pd_file.loc[n:, output_names[ind_out]].values
        self.out = torch.from_numpy(self.out).float()

    def __len__(self):
        return len(self.out)

    def __getitem__(self, idx):
        x = self.regressor[idx, :]
        y = self.out[idx].reshape(-1)

        return x, y


class F16DS_seq(Dataset):

    def __init__(self, pd_file, seq_len, ind_out):
        output_names = ['Acceleration1', 'Acceleration2', 'Acceleration3']
        self.pd_file = pd_file  # pandas dataset
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))
        u = np.zeros((n, seq_len, 2))

        out_list = [output_names[i] for i in ind_out]
        y = np.zeros((n, seq_len, len(out_list)))

        for ind in range(n):
            if ind<n-1:

                u[ind, :, 0] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len]['Force']

                u[ind, :, 1] = pd_file.iloc[ind * seq_len:(ind + 1) * seq_len]['Voltage']

                y[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len] [out_list]

            else:
                u[ind, :, 0] = pd_file.iloc[N - seq_len:]['Force']
                u[ind, :, 1] = pd_file.iloc[N - seq_len:][ 'Voltage']
                y[ind, :, :] = pd_file.iloc[N - seq_len:][ out_list]

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y


class WH2009_seq(Dataset):

    def __init__(self, pd_file, seq_len, ind_out = None):

        self.pd_file = pd_file  # pandas dataset
        print(pd_file)
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))
        print(n)
        u = np.zeros((n, seq_len, 1))

        y = np.zeros((n, seq_len, 1))


        for ind in range(n):
            if ind<n-1:
                print(ind)

                u[ind, :, 0] = pd_file.loc[ind*seq_len:(ind+1)*seq_len-1]['uBenchMark']

                y[ind, :, 0] = pd_file.loc[ind*seq_len:(ind+1)*seq_len-1] ['yBenchMark']

            else:
                u[ind, :, 0] = pd_file.loc[N - seq_len:]['uBenchMark']
                y[ind, :, 0] = pd_file.loc[N - seq_len:]['yBenchMark']

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y



class CED_seq(Dataset):
    def __init__(self, pd_file, seq_len, ind_out):
        output_names = ['z1', 'z2', 'z3']
        input_names = ['u1', 'u2', 'u3']
        self.pd_file = pd_file  # pandas dataset
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))

        in_list = [input_names[i] for i in ind_out]
        out_list = [output_names[i] for i in ind_out]
        u = np.zeros((n, seq_len, len(in_list)))
        y = np.zeros((n, seq_len, len(out_list)))

        for ind in range(n):
            if ind<n-1:

                u[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len][in_list]

                y[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len] [out_list]

            else:
                u[ind, :, :] = pd_file.iloc[N - seq_len:][in_list]
                y[ind, :, :] = pd_file.iloc[N - seq_len:][ out_list]

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y



class CT_seq(Dataset):
    def __init__(self, pd_file, seq_len):

        self.pd_file = pd_file  # pandas dataset
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))

        u = np.zeros((n, seq_len, 1))
        y = np.zeros((n, seq_len, 1))

        in_list = ['uEst']
        out_list = ['yEst']

        for ind in range(n):
            if ind<n-1:

                u[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len][in_list]

                y[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len] [out_list]

            else:
                u[ind, :, :] = pd_file.iloc[N - seq_len:][in_list]
                y[ind, :, :] = pd_file.iloc[N - seq_len:][ out_list]

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y

class SB_seq(Dataset):
    def __init__(self, pd_file, seq_len):


        self.pd_file = pd_file  # pandas dataset
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))

        u = np.zeros((n, seq_len, 1))
        y = np.zeros((n, seq_len, 1))

        in_list = ['V1']
        out_list = ['V2']

        for ind in range(n):
            if ind<n-1:

                u[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len][in_list]

                y[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len] [out_list]

            else:
                u[ind, :, :] = pd_file.iloc[N - seq_len:][in_list]
                y[ind, :, :] = pd_file.iloc[N - seq_len:][ out_list]

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y



class SB_MS_seq(Dataset):
    def __init__(self, pd_file, seq_len):


        self.pd_file = pd_file  # pandas dataset
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))

        u = np.zeros((n, seq_len, 1))
        y = np.zeros((n, seq_len, 1))

        in_list = ['V1']
        out_list = ['V2']

        for ind in range(n):
            if ind<n-1:

                u[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len][in_list]

                y[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len] [out_list]

            else:
                u[ind, :, :] = pd_file.iloc[N - seq_len:][in_list]
                y[ind, :, :] = pd_file.iloc[N - seq_len:][ out_list]

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y



class EMPS_seq(Dataset):
    def __init__(self, pd_file, seq_len):


        self.pd_file = pd_file  # pandas dataset
        self.seq_len = seq_len

        N = len(pd_file)
        n = int(np.ceil(len(pd_file)/seq_len))

        u = np.zeros((n, seq_len, 1))
        y = np.zeros((n, seq_len, 1))

        in_list = ['vir']
        out_list = ['qm']

        for ind in range(n):
            if ind<n-1:

                u[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len][in_list]

                y[ind, :, :] = pd_file.iloc[ind*seq_len:(ind+1)*seq_len] [out_list]

            else:
                u[ind, :, :] = pd_file.iloc[N - seq_len:][in_list]
                y[ind, :, :] = pd_file.iloc[N - seq_len:][ out_list]

        self.u = torch.from_numpy(u).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return  self.y.shape[0]

    def __getitem__(self, idx):
        u = self.u[idx, :, :]
        y = self.y[idx, :, :]

        return u, y





def create_dataset(benchmark, config, file_map, Normalization = True):


    if benchmark in config and benchmark in file_map:
        ds_class = config[benchmark]['class']
        params = config[benchmark]['params']
        f_train_ds = file_map[benchmark]

        #f_train_ds = os.path.join(folder, file)


        # Determine file type and load data accordingly
        if f_train_ds.endswith('.csv'):
            pd_file = pd.read_csv(f_train_ds)
        elif f_train_ds.endswith('.mat'):
            mat_data = loadmat(f_train_ds)  # Load .mat file

            if benchmark == 'EMPS':
                vir = mat_data['vir']  # Extract 'vir' data
                qm = mat_data['qm']  # Extract 'qm' data

                # Convert to DataFrame
                # This example assumes 'vir' and 'qm' are 2D arrays with compatible shapes for side-by-side concatenation
                data = np.hstack([vir, qm])  # Adjust as necessary based on your data structure
                column_names = ['vir', 'qm']
            pd_file = pd.DataFrame(data, columns=column_names)


        else:
            raise ValueError("File format not supported")



        # Load data
        #pd_file = pd.read_csv(f_train_ds)

        if Normalization ==True:
            pd_file = (pd_file - pd_file.mean()) / pd_file.std()
            return ds_class(pd_file=pd_file, **params), pd_file.mean(), pd_file.std()
        else:
            return ds_class(pd_file=pd_file, **params), 0, 1

    else:
        raise ValueError("Benchmark not supported")



if __name__ == "__main__":

    # Maps of benchmarks
    config = {
        'CED': {'class': CED_seq, 'params': {'seq_len': 100, 'ind_out': [0, 1, 2]}},
        'WH2009': {'class': WH2009_seq, 'params': {'seq_len': 188000 - 4}},
        'F16': {'class': F16DS_seq, 'params': {'seq_len': 24, 'ind_out': [0]}},
        'SB': {'class': SB_seq, 'params': {'seq_len': 131072}},
        'SB_MS': {'class': SB_MS_seq, 'params': {'seq_len': 131072}},
        'CT': {'class': CT_seq, 'params': {'seq_len': 1024}},
        'EMPS': {'class': EMPS_seq, 'params': {'seq_len': 24841}},
    }

    file_map = {
        'CED': '..\\Datasets\\CED\\DATAPRBS.csv',
        'WH2009': '..\\Datasets\\WH2009\\WienerHammerBenchmark.csv',
        'F16': '..\\Datasets\\F16\\F16Data_SineSw_Level3.csv',
        'SB': os.path.join('..', 'Datasets', 'Silverbox', 'SNLS80mV.csv'),
        'SB_MS': os.path.join('..', 'Datasets', 'Silverbox', 'Schroeder80mV.csv'),
        'CT': os.path.join('..','Datasets','CascadedTanks','dataBenchmark.csv'),
        'EMPS': os.path.join('..', 'Datasets', 'EMPS', 'DATA_EMPS.mat')
    }



    benchmark = 'SB_MS'  # or 'CED', 'F16'
    ds_train, mean, std = create_dataset(benchmark, config, file_map,  Normalization=True)

    loader_train = DataLoader(ds_train, shuffle=True, batch_size=2)
    u, y = next(iter(loader_train))
    print(u.shape, y.shape)
    plt.figure()
    plt.plot(y[0,:,0])
    plt.show()


    #ind_out = 0 # output index (0,1,2)
    #na = 5 # output lag
    #nb = 20 # input lag
    #F16DS_train = F16DS(pd_file=dict_ds['train'], na=na, nb=nb, ind_out=ind_out)
    #loader_train = DataLoader(F16DS_train, shuffle=True, batch_size=1)
    #x, y = next(iter(loader_train))
    #print(x.shape, y.shape)


    #F16DS_train = F16DS_seq(pd_file=dict_ds['train'][55000:70000], seq_len = 23, ind_out = [0])
    #loader_train = DataLoader(F16DS_train, shuffle=True, batch_size=2)
    #u, y = next(iter(loader_train))
    #print(u.shape, y.shape)

    #WH2009_train = WH2009_seq(pd_file=dict_ds['train'], seq_len = 188000-4)


