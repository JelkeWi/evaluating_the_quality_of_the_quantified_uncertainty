import sys
import pandas as pd
import logging
from torch.utils.data import  Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pathlib



class Custom_Dataset(Dataset):
    def __init__(self, path, yfeature: str, random_state: int, include_feature: list = [], exclude_feature: list = [], categorical_feature: list = [], test_split: float = 0.2, 
                 val_split: float = 0.0, scaler: str | None = 'minmax', limit_nr_samples: int = 0, one_hot_encoding: bool = False):
        # init from args
        self.path = path
        self.yfeature = yfeature
        self.categorical_feature = categorical_feature
        self.random_state = random_state
        self.test_split = test_split
        self.val_split = val_split
        self.train_split = 1 - test_split - val_split
        self.scaler = scaler
        self.limit_nr_samples = limit_nr_samples # limit the number of samples if != 0.
        self.one_hot_encoding = one_hot_encoding
        
        #init
        self.num_yfeature = len([yfeature]) #default is 1, as yfeature is a string
        self.num_categorical_feature = len(categorical_feature)
        # init placeholder
        self.data = pd.DataFrame()
        self.num_samples = int()
        self.xfeature = list()
        self.num_xfeature = int()
        self.num_encoded_feature = int()
        self.scaler_dict = dict()
        self.encoder_dict = dict()

        #functions
        self.normalize_data = normalize_data

        # load data
        self._load_data(include_feature, exclude_feature)
        # preprocess data
        self._preprocessing()


    def __len__(self):
        return len(self.data)

    def __getitems__(self, idx): # function for getting a single sample
        # print('here')
        batch = self.data.iloc[idx]
        seq = batch[self.xfeature].to_numpy()
        label = batch[self.yfeature].to_numpy()
        mask = batch.index.to_numpy()
        return seq, label, mask


    def _load_data(self, include_feature: list, exclude_feature: list):
        # load dataset using the included features list, or dropping the excluded feature list
        if include_feature and exclude_feature:
            sys.exit('Error: In Dataset {} features are included and excluded, which is mutually exclusive.\n Included: {}\n Excluded: {}'.format(self.path, include_feature, exclude_feature))

        suffix = pathlib.Path(self.path).suffix[1:]
        if suffix == 'csv':
            if include_feature:
                # load dataset only with included features
                data = pd.read_csv(self.path, usecols= include_feature + [self.yfeature], sep =None, engine='python')
            else:
                data = pd.read_csv(self.path, sep =None, engine='python')
        elif suffix == 'xls' or suffix == 'xlsx':
            if include_feature:
                data = pd.read_excel(self.path, usecols= include_feature + [self.yfeature])
            else:
                data = pd.read_excel(self.path)
        else:
            sys.exit('No possibility to open \'{}\' files implemented. File path: {}'.format(suffix, self.path))

        if 'Unnamed: 0' in data.columns: # drop pandas specific indexing column
            data = data.drop('Unnamed: 0', axis = 1)
        data = data.dropna(axis=1, how='all')
        for feat in exclude_feature:
            if feat not in data.columns:
                logging.warning(f'Feat {feat=} is part of {exclude_feature=} but could not be found in {data.columns=} of dataset {self.path=}')
        exclude_col = list(~data.columns.isin(exclude_feature))
        self.data = data.loc[:, exclude_col]

        
    def _preprocessing(self):

        # drop nan samples
        self.data = self.data.dropna()

        if self.limit_nr_samples != 0: #limit the number of samples
            if self.limit_nr_samples < len(self.data):
                self.data = self.data.sample(n = self.limit_nr_samples, random_state= self.random_state)
            else:
                print(f'Imported dataset {self.path} could not be limited to {self.limit_nr_samples} samples as it only contains {len(self.data)}.')
        self.num_samples = len(self.data)
        
        # encode categorical feature
        if len(self.categorical_feature) == 0: # check for categorical features
            are_all_numeric = all(self.data.select_dtypes(include=['number']).columns == self.data.columns)
            if are_all_numeric == False:
                logging.warning('Not all features are numberic, but no categorical features where handed over when initialising the dataset.')
        else:
            # convert ints that are stored as float to int. Otherwise the pd.dummies feature names contain floats
            numeric_columns = self.data.select_dtypes(include=['float']).columns
            self.data[numeric_columns] = self.data[numeric_columns].map(lambda x: int(x) if x.is_integer() else x) 
            if self.one_hot_encoding == True:
                logging.debug('Applying One-Hot-Encoding...')
                new_cat_feature = []
                for cat_feature in self.categorical_feature:
                    # convert binary features to categorical
                    if self.data[cat_feature].nunique() == 2:
                        dummies = pd.get_dummies(self.data[cat_feature], prefix=cat_feature, drop_first= True)
                    # one-hot-encode the rest of the features
                    else:
                        dummies = pd.get_dummies(self.data[cat_feature], prefix=cat_feature)
                    self.data = self.data.drop(cat_feature, axis=1)
                    self.data = pd.concat([self.data, dummies], axis=1)
                    new_cat_feature.extend(list(dummies.columns)) # save expanded feature
                self.categorical_feature = new_cat_feature
            else:
                self.encode_labels(self.categorical_feature)
        
        xfeature = set(self.data.columns)
        xfeature.remove(self.yfeature)
        self.xfeature = list(xfeature)
        self.num_xfeature = len(self.xfeature)
        self.num_categorical_feature = len(self.categorical_feature)

        # normalize data
        self.data, self.scaler_dict = self.normalize_data(self.data, self.scaler)
        numeric_columns = self.data.select_dtypes(include=['float']).columns
        self.data[numeric_columns] = self.data[numeric_columns].map(lambda x: int(x) if x.is_integer() else x) 
        self.data[self.categorical_feature] = self.data[self.categorical_feature].astype('category')
        

    def encode_labels(self, cat_features):
        # encode non-numeric/categorical features
        encoder_dict = {}
        for col in cat_features:
            encoder_dict[col] = LabelEncoder() #init encoder 
            encoder_dict[col] = encoder_dict[col].fit(self.data[col]) # fit the encoder to data
            self.data[col] = encoder_dict[col].transform(self.data[col].to_numpy()) # encode data, has to be transformed, .transform() does not like series
            self.data[col] = self.data[col].astype('category') # can maybe left out.
        self.encoder_dict = encoder_dict
        logging.debug("Encode categorical features: done")



def standard_scale_data(data: pd.DataFrame):
    scaler_dict = {} #dict to store scalers
    for col in data.columns:
        scaler_dict[col] = StandardScaler() #init scaler 
        scaler_dict[col] = scaler_dict[col].fit(data[col].to_numpy().reshape(-1, 1)) # fit the scaler to data
        data[col] = scaler_dict[col].transform(data[col].to_numpy().reshape(-1, 1)) #scale data
    scaler_dict = scaler_dict
    logging.debug("Standard scale: done")
    return data, scaler_dict



def minmax_scale_data(data: pd.DataFrame):
    scaler_dict = {} #dict to store scalers
    for col in data.columns:
        scaler_dict[col] = MinMaxScaler() #init scaler 
        scaler_dict[col] = scaler_dict[col].fit(data[col].to_numpy().reshape(-1, 1)) # fit the scaler to data
        data[col] = scaler_dict[col].transform(data[col].to_numpy().reshape(-1, 1)) #scale data
    scaler_dict = scaler_dict
    logging.debug("Minmax scale: done")
    return data, scaler_dict


def normalize_data(data: pd.DataFrame, config: str | None):
    if config == 'minmax':
        data, scaler_dict = minmax_scale_data(data)
    elif config == 'standard':
        data, scaler_dict = standard_scale_data(data)
    elif config == None:
        scaler_dict = {}
    else:
        sys.exit(f'Error: {config=}. No suitable normalization mode found')
    return data, scaler_dict