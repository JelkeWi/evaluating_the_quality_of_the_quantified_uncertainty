import pandas as pd
import numpy as np
from typing import Literal
from utils.dataset import Custom_Dataset
from pyod.models.lof import LOF
from pyod.models.iforest import IForest


def remove_outlier_and_split_dataset(
        dataset_path: str,
        yfeature: str,
        exclude_feature: list,
        categorical_feature: list,
        random_state: int,
        scaler: str|None = None,
        method: Literal['LOF', 'IForest'] = 'IForest',
        probability_threshold: float = 0.8
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    
    np.random.seed(random_state)

    new_dataset_properties = pd.Series(index = ['n_feature', 'n_sample', 'n_cleaned_anomalies' , 'target', 'exclude', 'categorical'], dtype= object)


    dset = Custom_Dataset(path = dataset_path, yfeature= yfeature, random_state= random_state, exclude_feature= exclude_feature, categorical_feature= categorical_feature,
                        scaler= scaler)
    
    # Adjust data types
    for col in dset.data.columns:
        if col in categorical_feature:
            dset.data[col] = dset.data[col].astype('category')
    
    # Clean data
    __, outlier_labels = extract_outliers(dset.data, method, probability_threshold= probability_threshold, random_state =  random_state)

    data = dset.data[outlier_labels == False]
    data = data.reset_index(drop = True)
    data = data[[yfeature] + [col for col in data.columns if col != yfeature]]
    new_dataset_properties.loc['n_feature'] = len(data.columns)
    new_dataset_properties.loc['n_sample'] = len(data)
    new_dataset_properties.loc['n_cleaned_anomalies'] = int(outlier_labels.sum())
    new_dataset_properties.loc['target'] = yfeature
    if len(categorical_feature) > 0:
        new_dataset_properties.loc['categorical'] = ', '.join(dset.categorical_feature)
    
    # split dataset 
    train, test = np.split(data.sample(frac=1, random_state = random_state), [int(.7*len(data))])
    train = np.array(train)
    test = np.array(test)
    train = pd.DataFrame(train, columns = data.columns)
    test = pd.DataFrame(test, columns = data.columns)
    
    return data, train, test, new_dataset_properties


def extract_outliers(data: pd.DataFrame, method: Literal['LOF', 'IForest'], probability_threshold: float = 0.9, random_state: int = 0) -> tuple:
    """
    Extract outliers from the given data using the specified outlier detection method.

    Parameters:
    - data (pd.DataFrame): The input data, where rows are samples and columns are features.
    - method (Literal['LOF', 'IForest']): The outlier detection method to use ('LOF' for Local Outlier Factor, 'IForest' for Isolation Forest).
    - probability_threshold (float, optional): The probability threshold for classifying samples as outliers. Defaults to 0.9.

    Returns:
    tuple: A tuple containing:
        - probability_outlier (pd.Series): A series containing outlier probabilities for each sample.
        - outlier_labels (pd.Series): A boolean series indicating whether each sample is an outlier based on the probability threshold.
    """
    if method == "IForest":
        clf = IForest(random_state=random_state)
    elif method == "LOF":
        clf = LOF()
    else:
        sys.exit(f'Wrong outlier method: {method=}')
    
    clf.fit(data.values)
    probability = clf.predict_proba(data.values, return_confidence=False)
    probability_outlier = probability[:, 1] # type: ignore

    probability_outlier = pd.Series(index=data.index, data=probability_outlier)
    probability_outlier.name = 'probability'
    outlier_labels = probability_outlier > probability_threshold
    outlier_labels.name = 'is_outlier'

    return probability_outlier, outlier_labels


def get_values_from_dataset_table(dataset_df, dname):
    """
    Extracts target feature, features to exclude, and categorical features 
    from a dataset table based on the given dataset name.

    This function retrieves specific information from a DataFrame (`dataset_df`) 
    that contains dataset configuration details. It looks up the row corresponding 
    to the dataset name (`dname`) and extracts the target feature, a list of 
    features to exclude, and a list of categorical features.

    Args:
        dataset_df (pandas.DataFrame): A DataFrame containing dataset configuration details. 
                                       The DataFrame should have columns named 'target', 
                                       'exclude', and 'categorical'.
        dname (str): The name of the dataset to retrieve configuration details for.

    Returns:
        tuple: A tuple containing:
            - yfeature (str): The target feature for the dataset.
            - exclude_feature (list of str): List of features to exclude, or an empty list if none are specified.
            - categorical_feature (list of str): List of categorical features, or an empty list if none are specified.

    Raises:
        KeyError: If `dname` is not found in the DataFrame index.
        ValueError: If the DataFrame does not contain required columns.
    """


    yfeature = str(dataset_df.loc[dname, 'target'])
    exclude_feature = dataset_df.loc[dname, 'exclude']
    if isinstance(exclude_feature, str):
        exclude_feature = [item.strip() for item in exclude_feature.split(',')]
    else:
        exclude_feature = []
    categorical_feature = dataset_df.loc[dname, 'categorical']
    if isinstance(categorical_feature, str):
        categorical_feature =  [item.strip() for item in categorical_feature.split(',')] # split at comma an strip spaces
    else:
        categorical_feature =  []
    return yfeature, exclude_feature, categorical_feature