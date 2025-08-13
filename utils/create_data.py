import pandas as pd
import os, logging
import sys
import ucimlrepo
from utils.synthetic_data_generation import make_nernst, make_euclidean, make_stribeck, make_arctan, make_mlp, make_sklearn
from utils.data_preparation import get_values_from_dataset_table



def generate_synthetic_data(num_samples: int = 1000, noise_list: list = [0.0, 0.03]):
    """Create various synthetic datasets with or without noise.

    Args:
        num_samples (int, optional): Size of the datasets. Defaults to 1000.
        noise_list (list, optional): Noise standard deviation added to the dataset. One dataset is created for each level of noise in the list. Defaults to [0.0, 0.03].
    """
    dataset_df = pd.DataFrame(columns=['name', 'n_feature', 'n_sample', 'target', 'exclude', 'categorical'])
    base_folder = os.getcwd()
    data_path = os.path.join(base_folder, "data", "synthetic")
    logging.info('Begin creating the synthetic data sets')
    # set name prefix in case of noise
    for noise in noise_list:
        name_prefix = 'noise_' + str(noise).replace('.', '_') + '_'

        data, metrics = make_euclidean(num_samples= num_samples,
                    noise= noise,
                    path=data_path, name_prefix= name_prefix)
        dataset_df = pd.concat([dataset_df, metrics.to_frame().T])

        data, metrics = make_nernst(num_samples=num_samples,
                                    noise=noise,
                                    path=data_path, name_prefix= name_prefix)
        dataset_df = pd.concat([dataset_df, metrics.to_frame().T])

        data, metrics = make_stribeck(num_samples= num_samples,
                                noise= noise,
                                path=data_path, name_prefix= name_prefix)
        dataset_df = pd.concat([dataset_df, metrics.to_frame().T])

        data, metrics = make_arctan(num_samples= num_samples,
                                noise= noise,
                                path=data_path, name_prefix= name_prefix)
        dataset_df = pd.concat([dataset_df, metrics.to_frame().T])

        data, metrics = make_mlp(num_samples= num_samples,
                                noise= noise,
                                path=data_path, name_prefix= name_prefix)
        dataset_df = pd.concat([dataset_df, metrics.to_frame().T])

        sklearn_sets = ['make_regression', 'make_sparse_uncorrelated', 'make_friedman1', 'make_friedman2', 'make_friedman3']
        for i in sklearn_sets:
            data, metrics = make_sklearn(function_name=i,
                                        num_samples= num_samples,
                                        noise=noise,
                                        path=data_path,
                                        name=i, name_prefix= name_prefix)
            dataset_df = pd.concat([dataset_df, metrics.to_frame().T])
    
    dataset_df['No.'] = range(len(dataset_df))
    dataset_df = dataset_df.set_index('No.', drop = True)
    dataset_df.to_excel(data_path + '.xlsx', index= True)
    logging.info(f'Succesfully created {len(dataset_df)} synthetic data sets.')



def fetch_real_world_data(dataset_df: pd.DataFrame):
    """Download and save data from the UCIML repository.
    The name of the dataset has to be the index of dataset_df.
    Args:
        dataset_df (pd.DataFrame): Dataframe containing the dataset IDs.
    """
    data_path = os.path.join(os.getcwd(), "data", "real")
    os.makedirs(data_path, exist_ok=True)
    for idx, series in dataset_df.iterrows():
        print(f'Processing dataset: {idx} (ID={series['ID']})')
        try:
            data_raw = ucimlrepo.fetch_ucirepo(id = series['ID'])
            data_df = data_raw['data']['original']    
            data_df.to_csv(os.path.join(data_path, str(idx)) + '.csv', index_label = False)
            print(f'Downloaded and saved dataset {idx} containing {len(data_df)} samples.')
        except Exception as e:
            print(e)
            sys.exit()
        
        # check if the dataset_df was/is still correct.
        yfeature, exclude_feature, categorical_feature = get_values_from_dataset_table(dataset_df, idx)
        try:
            data_df[yfeature]
        except Exception as e:
            print(f'Could not find target feature {yfeature} in the dataset. Most likely a typo somewhere.', e)
        try:
            data_df[exclude_feature]
        except Exception as e:
            print(f'Could not find exclude_feature feature {exclude_feature} in the dataset. Most likely a typo somewhere.', e)
        try:
            data_df[categorical_feature]
        except Exception as e:
            print(f'Could not find categorical_feature feature {categorical_feature} in the dataset. Most likely a typo somewhere.', e)
        
        dataset_df.loc[str(idx), 'n_feature'] = len(data_df.columns) -1 
        dataset_df.loc[str(idx), 'n_sample'] = len(data_df)
    dataset_df.reset_index().set_index('No.').to_excel(data_path + '.xlsx', index= True) # Save the df in that No. is the index.
    logging.info('Finished downloading and saving the real world data sets.')