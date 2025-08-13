import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from sklearn.datasets import make_regression, make_friedman1, make_friedman2, make_friedman3, make_sparse_uncorrelated

def make_euclidean(num_samples, noise, path, name = 'euclidean_distance', ylower_bound = 2, yupper_bound = 10, random_state = 1, name_prefix = ''):
    '''Generate dataset using the euclidean distance. The data is generated based on polar coordinates to make y gausian distributed.'''
    np.random.seed(random_state)
    # generate base dataset to sample from
    base_data = pd.DataFrame(columns = ['x1', 'x2', 'y'], index = pd.RangeIndex(stop = num_samples))
    target = np.random.normal(size = num_samples)
    base_data['y'] = normalize_array(target, ylower_bound, yupper_bound)
    base_data['angle'] = np.random.uniform(low = 0, high = 2*np.pi, size = num_samples)
    base_data['x1'] = np.sin(base_data['angle'])*base_data['y']
    base_data['x2'] = np.cos(base_data['angle'])*base_data['y']
    # get only the relevant features
    data = pd.DataFrame(base_data[['y', 'x1', 'x2']])
    # apply noise
    data = apply_noise(data, noise)
    # save and report metrics
    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, name_prefix + name) + '.csv', index_label = False)
    metrics = pd.Series(index = ['name', 'n_feature', 'n_sample', 'target'], data = [name_prefix+name, 2, len(data), 'y'])
    return data, metrics

def apply_noise(data, noise):
    for col in data.columns:
        # delta = data[col].max() - data[col].min()
        std = data[col].std()
        noise_scale = std * noise
        data[col] = data[col] + np.random.normal(scale = noise_scale, size=len(data[col]))
    return data


def make_nernst(num_samples, noise, path, name = 'nernst', random_state = 1, name_prefix = ''):
    '''Generate a dataset using the cell potential fo the nernst equation'''

    np.random.seed(random_state)
    R = np.random.uniform(low = 1, high = 10, size = num_samples*2)
    T = np.random.uniform(low = 1, high = 10, size = num_samples*2)
    z = np.random.uniform(low = 1, high = 10, size = num_samples*2)
    F = np.random.uniform(low = 1, high = 10, size = num_samples*2)
    a_r = np.random.uniform(low = 1, high = 10, size = num_samples*2)
    a_o = np.random.uniform(low = 1, high = 10, size = num_samples*2)

    # cell potential
    E = (R * T) / (z*F) * np.log(a_r/a_o)

    base_data = pd.DataFrame(data = np.column_stack((T, z, F, a_r, a_o, E)), columns= ['T', 'z', 'F', 'a_r', 'a_o', 'E'])

    lower_bound = base_data['E'].quantile(0.05)
    upper_bound = base_data['E'].quantile(0.95)
    data = base_data[(base_data['E'] > lower_bound) & (base_data['E'] < upper_bound)]
    data = data.sample(n = num_samples)

    data = apply_noise(data, noise)
    # save and report metrics
    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, name_prefix+name) + '.csv', index_label = False)
    metrics = pd.Series(index = ['name', 'n_feature', 'n_sample', 'target'], data = [name_prefix+name, len(data.columns)-1, len(data), 'E'])
    return data, metrics


def make_arctan(num_samples, noise, path, name = 'arctan', random_state = 1, name_prefix = ''):
    '''Create dataset using the arctan function'''
    np.random.seed(random_state)

    y = np.random.normal(loc = 0, scale = np.pi, size=num_samples)
    data = pd.DataFrame()
    data['y'] = y
    data['x'] = np.arctan(y)

    data = apply_noise(data, noise)
    # save and report metrics
    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, name_prefix+name) + '.csv', index_label = False)
    metrics = pd.Series(index = ['name', 'n_feature', 'n_sample', 'target'], data = [name_prefix+name, len(data.columns)-1, len(data), 'y'])
    return data, metrics


def normalize_array(arr, lower_bound, upper_bound):
    '''MinMax normalize an array.'''
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    # Normalize to [0, 1]
    normalized = (arr - arr_min) / (arr_max - arr_min)
    
    # Scale to [lower_bound, upper_bound]
    scaled = normalized * (upper_bound - lower_bound) + lower_bound
    
    return scaled


def make_stribeck(num_samples, noise, path, name = 'stribeck', random_state = 1, name_prefix = ''):
    """Generate a synthetic dataset based on the Striebeck equation."""

    np.random.seed(random_state)

    u_g = np.random.chisquare(df = 3, size= num_samples)
    u_g = (u_g - u_g.min()) / (u_g.max() - u_g.min()) * (1 - 0.01) + 0.01

    u_h = np.random.weibull(3, num_samples)
    u_h = (u_h - u_h.min()) / (u_h.max() - u_h.min()) * (1 - 0.01) + 0.01

    F_N = np.random.triangular(1, 6, 8, num_samples)
    F_N_min = 100*np.cos(2*np.pi / 12)*9.81 # 100 kg on a 30 degree slope
    F_N_max = 100 * 9.81  # 100 kg on a 0 degree slope
    F_N = (F_N - F_N.min()) / (F_N.max() - F_N.min()) * (F_N_max - F_N_min) + F_N_min

    v = -np.random.rayleigh(5, num_samples)
    v = (v - v.min()) / (v.max() - v.min()) * (100 - 0.01) + 0.01

    v_s = np.random.power(6, num_samples)
    v_s = (v_s - v_s.min()) / (v_s.max() - v_s.min()) * (1 - 0.01) + 0.01

    d = np.random.wald(1, 2, num_samples)
    d = (d - d.min()) / (d.max() - d.min()) * (3 - 0.01) + 0.01

    F_stri = u_g * F_N + (u_h * F_N - u_g * F_N) * np.exp(- abs(v/v_s)**d)
    data = pd.DataFrame(data = np.column_stack((u_g, u_h, F_N, v, v_s, d, F_stri)), columns= ['u_g', 'u_h', 'F_N', 'v', 'v_s', 'd', 'F_stri'])

    data = apply_noise(data, noise)
    # save and report metrics
    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, name_prefix+name) + '.csv', index_label = False)
    metrics = pd.Series(index = ['name', 'n_feature', 'n_sample', 'target'], data = [name_prefix+name, len(data.columns)-1, len(data), 'F_stri'])
    return data, metrics


def make_mlp(num_samples, noise, path, name = 'mlp', random_state = 1, num_xfeature = 4, name_prefix = ''):
    '''Create a dataset using a multilayer perceptron'''
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # get mlp model
    layer_size = 128
    num_layers = 2
    model = get_simple_MLP(num_xfeature, 1, [layer_size for i in range(num_layers)], [0.0 for i in range(num_layers)])

    xdata = torch.DoubleTensor(np.random.normal(size = (int(num_samples), num_xfeature)))
    data = pd.DataFrame(data = xdata.numpy(), index = range(xdata.shape[0]), columns= [f'x{i}' for i in range(xdata.shape[1])])

    ydata = model(xdata)    
    data['y'] = ydata.view(-1).detach().numpy()

    data = apply_noise(data, noise)
    # save and report metrics
    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, name_prefix+name) + '.csv', index_label = False)
    metrics = pd.Series(index = ['name', 'n_feature', 'n_sample', 'target'], data = [name_prefix+name, len(data.columns)-1, len(data), 'y'])
    return data, metrics

def make_sklearn(function_name, num_samples, noise, path, name, random_state = 1, name_prefix = ''):
    match function_name:
        case 'make_regression':
            x, y, coef = make_regression(n_samples= num_samples, n_features= 3, coef= True, random_state= random_state)
        case 'make_sparse_uncorrelated':
            x, y = make_sparse_uncorrelated(n_samples=num_samples, n_features= 4, random_state= random_state)
        case 'make_friedman1':
            x, y = make_friedman1(n_samples=num_samples, n_features= 5, random_state= random_state)
        case 'make_friedman2':
            x, y = make_friedman2(n_samples=num_samples, random_state= random_state)
        case 'make_friedman3':
            x, y = make_friedman3(n_samples=num_samples, random_state= random_state)
        case _:
            raise ValueError(f'Got function name {function_name}, but that does not exist.')


    data = pd.DataFrame(x, columns= [f'x{i}' for i in range(x.shape[1])])
    data['y'] = y      
    data = apply_noise(data, noise)
    # save and report metrics
    os.makedirs(path, exist_ok=True)
    data.to_csv(os.path.join(path, name_prefix+name) + '.csv', index_label = False)
    metrics = pd.Series(index = ['name', 'n_feature', 'n_sample', 'target'], data = [name_prefix+name, len(data.columns)-1, len(data), 'y'])
    return data, metrics



def get_simple_MLP(num_xfeatures :int, num_yfeatures :int, layer_size :list, dropout :list) -> nn.Module:
    class feed_forward_MLP(nn.Module):
        def __init__(self):
            super(feed_forward_MLP, self).__init__()
            #self.flatten = nn.Flatten()
            self.model_stack = nn.Sequential()
            self.model_stack.append(nn.Linear(num_xfeatures, layer_size[0]))
            if dropout[0] > 0.0:
                self.model_stack.append(nn.Dropout(p=dropout[0], inplace=False))
            for i in range(1, len(layer_size)):
                self.model_stack.append(nn.Linear(layer_size[i-1], layer_size[i]))
                self.model_stack.append(nn.ReLU())
                if dropout[i] > 0.0:
                    self.model_stack.append(nn.Dropout(p=dropout[i], inplace=False))
            self.model_stack.append(nn.Linear(layer_size[-1], num_yfeatures))


        def forward(self, x):
            logits = self.model_stack(x)
            return logits
        
    model = feed_forward_MLP().to("cpu")
    model = model.double()
    return model