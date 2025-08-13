'''
Script to run the miscalibration case studies of the benchmark.
Author: Jelke Wibbeke, Nico Schönfisch
'''
import logging, argparse, os, sys, logging, warnings
import pandas as pd
import numpy as np
from utils.data_preparation import remove_outlier_and_split_dataset, get_values_from_dataset_table
from utils.create_data import generate_synthetic_data, fetch_real_world_data
from utils.nll_ensemble_trainer import eval_ensemble
from utils.misc import concat_excel_files

def run_datasets(dataset_df, data_dir: str, result_dir: str, random_state: int, test_mode: bool, scenario_name: str, max_run: int):
    """Train ensembles for all datasets listed in the dataset_df and evaluate the ensemble performance on the metrics.

    Args:
        dataset_df (_type_): Dataframe containing names of the datasets to be run.
        data_dir (str): Data source directory.
        result_dir (str): Save file directory.
        random_state (int): Random state.
        test_mode (bool): Set to True for testing purposes. Limits the dataset size to 500 samples.
        scenario_name (str): Name of the run. Used for naming of save files.
    """
    # prepare data
    metrics_df = pd.DataFrame()
    for i, dname in enumerate(dataset_df.index):
        logging.info(f'Begin dataset: \'{dname}\' ({i+1}/{len(dataset_df)}). Run {random_state+1}/{max_run}.')
        metrics = pd.Series()
        metrics['name'] = dname
        metrics['n_run'] = random_state
        np.random.seed(random_state)
        # Get dataset
        yfeature, exclude_feature, categorical_feature = get_values_from_dataset_table(dataset_df, dname)
        dataset_path = os.path.join(data_dir, f'{dname}.csv')
        data, train, test, new_dataset_properties = remove_outlier_and_split_dataset(dataset_path= dataset_path,
                                        yfeature= yfeature,
                                        exclude_feature= exclude_feature,
                                        categorical_feature= categorical_feature,
                                        random_state= random_state,
                                        scaler= 'minmax')

        metrics['n_feature'] = new_dataset_properties.loc['n_feature']
        metrics['n_sample'] = new_dataset_properties.loc['n_sample']
        metrics['n_cleaned_anomalies'] = new_dataset_properties.loc['n_cleaned_anomalies']
        metrics['n_sample_test'] = len(test)
        metrics['n_sample_train'] = len(train)

        if test_mode:
            if len(train) > 500:
                train = train.sample(n=500, replace= False, random_state= random_state)
        logging.info(f'Finished data preparation...')

        # Generate miscalibrated data
        true_mean = data[yfeature].to_numpy()
        true_mean = np.sort(true_mean)
        target_range = max(true_mean) - min(true_mean)

        epistemic_std = 0.05 * target_range + np.sin(2 * np.pi * true_mean / target_range)**2
        aleatoric_noise = np.random.normal(loc=0.0, scale=0.05 * target_range, size=true_mean.shape[0])

        # Add aleatoric randomness and ensure total std is valid
        prediction_std = (epistemic_std + aleatoric_noise).clip(min=1e-6)
        prediction_mean = np.random.normal(loc=true_mean, scale=prediction_std)

        # Evaluate metrics
        metrics_eval = eval_ensemble(prediction_mean=prediction_mean,
                                    prediction_std=prediction_std,
                                    true_mean= true_mean,
                                    target_range = target_range)
        metrics_eval['calibration'] = 'optimal'
        
        # Introduce miscalibration and evaluate
        lower_std = eval_ensemble(prediction_mean=prediction_mean, prediction_std=prediction_std*0.9, true_mean=true_mean, target_range=target_range)
        lower_std['calibration'] = 'lower_std'
        
        lower_pred = eval_ensemble(prediction_mean=prediction_mean*0.9, prediction_std=prediction_std, true_mean=true_mean, target_range=target_range)
        lower_pred['calibration'] = 'lower_pred'
        
        h_wippe_std = eval_ensemble(prediction_mean=prediction_mean, prediction_std=prediction_std*np.linspace(0.9,1.1,true_mean.shape[0]), true_mean=true_mean, target_range=target_range)
        h_wippe_std['calibration'] = 'heterogeneous_std'

        h_wippe_brev = eval_ensemble(prediction_mean=prediction_mean*np.linspace(0.9,1.1,true_mean.shape[0]), prediction_std=prediction_std*np.linspace(1.1,0.9,true_mean.shape[0]), true_mean=true_mean, target_range=target_range)
        h_wippe_brev['calibration'] = 'heterogeneous_both'

        metrics_eval_df = pd.DataFrame([metrics_eval, lower_std, lower_pred, h_wippe_std, h_wippe_brev])
        for key, value in metrics.items():
            metrics_eval_df[key] = value


        # Reporting
        metrics_df = pd.concat([metrics_df, metrics_eval_df])
        metrics_df = metrics_df.reset_index(drop=True)
        metrics_df.to_excel(os.path.join(result_dir,f'metrics_{scenario_name}_{random_state}.xlsx'), index= True)
        logging.info(f'Finished training...')
        logging.info(f'Saved results to \'metrics_{scenario_name}_{random_state}.xlsx\'')

    logging.info(f"Run {random_state}: Training and evaluation of all datasets finished ({random_state+1}/{max_run}).")





if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%d/%m/%Y %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=int, choices = [10,20,30,40,50], default = 20,
                    help="Sets the logging level. All logging related to the set value or higher is shown. DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50. Default: 30")
    parser.add_argument("--test_mode", action="store_true", help= "Run the script in test mode. Reduces the number of samples and datasets.")
    parser.add_argument("--create_synthetic_data", action="store_true", help= "Create the synthetic data sets.")
    parser.add_argument("--create_real_data", action="store_true", help= "Fetch the real world data sets.")
    parser.add_argument("--run_synthetic_data", action="store_true", help= "Run the experiments using the synthetic data sets.")
    parser.add_argument("--run_real_data", action="store_true", help= "Run the experiments using the real world data sets.")
    parser.add_argument("--n_runs", type= int, default = 10, help= "Number of models per data set")
    parser.add_argument("--name", type= str, help= "Define name of the scenario.", required=True)
    args = parser.parse_args()
    logging.info('Setting log level to: %i ', int(args.log_level))
    logging.getLogger().setLevel(int(args.log_level))


    scenario_name = args.name
    n_runs = args.n_runs

    # generate data
    if args.create_synthetic_data:
        generate_synthetic_data(num_samples = 1000, noise_list =[0.0])
    if args.create_real_data:
        dataset_df = pd.read_excel(os.path.join(os.getcwd(), 'uciml.xlsx'), index_col= 1)
        fetch_real_world_data(dataset_df=dataset_df)

    # check if results already exist.
    if args.run_synthetic_data or args.run_real_data:
        result_dir = os.path.join(os.getcwd(), 'results', scenario_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        else:
            sys.exit(f'The result directory {result_dir} already exists. Delete the directory to start with clean directory, otherwise concatenating the result metrics can fail.')

        # run models
        if args.run_synthetic_data:
            logging.info('Begin running synthetic datasets...')
            data_dir = os.path.join(os.getcwd(), 'data','synthetic')
            dataset_df = pd.read_excel(data_dir + '.xlsx', index_col= 1)
            dataset_df = dataset_df.sort_values(by = 'n_sample')
            if args.test_mode:
                dataset_df = dataset_df.iloc[:3]
            logging.info(f"Use datasets: {dataset_df.index}")
            for run in range(n_runs):
                run_datasets(dataset_df= dataset_df,
                            data_dir= data_dir,
                            result_dir=result_dir,
                            random_state = run,
                            test_mode= args.test_mode,
                            scenario_name= 'sythetic',
                            max_run = n_runs)
            logging.info('Finished running synthetic datasets...')
        if args.run_real_data:
            logging.info('Begin running real world datasets...')
            data_dir = os.path.join(os.getcwd(), 'data','real')
            dataset_df = pd.read_excel(data_dir + '.xlsx', index_col= 1)
            dataset_df = dataset_df.sort_values(by = 'n_sample')
            if args.test_mode:
                dataset_df = dataset_df.iloc[:3]
            logging.info(f"Use datasets: {dataset_df.index}")
            for run in range(n_runs):
                run_datasets(dataset_df= dataset_df,
                            data_dir= data_dir,
                            result_dir=result_dir,
                            random_state = run,
                            test_mode= args.test_mode,
                            scenario_name= 'real',
                            max_run = n_runs)
            logging.info('Finished running real world datasets...')

        # Concatenate results.
        concat_excel_files(result_dir, keyword = 'metrics_', keep_originals=True)
