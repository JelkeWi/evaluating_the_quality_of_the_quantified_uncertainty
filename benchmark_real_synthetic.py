'''
Script to run the real-world data and synthetic data case studies of the benchmark.
Author: Jelke Wibbeke, Nico Schönfisch
'''
import logging, argparse, os, sys, logging, warnings
import pandas as pd
import numpy as np
from utils.data_preparation import remove_outlier_and_split_dataset, get_values_from_dataset_table
from utils.create_data import generate_synthetic_data, fetch_real_world_data
from utils.nll_ensemble_trainer import train_and_eval_nll_ensemble
from utils.misc import concat_excel_files

def run_datasets(dataset_df, data_dir: str, result_dir: str, random_state: int, test_mode: bool, ensemble_size: int, scenario_name: str, max_run: int, recalibration: bool):
    """Train ensembles for all datasets listed in the dataset_df and evaluate the ensemble performance on the metrics.

    Args:
        dataset_df (_type_): Dataframe containing names of the datasets to be run.
        data_dir (str): Data source directory.
        result_dir (str): Save file directory.
        random_state (int): Random state.
        test_mode (bool): Set to True for testing purposes. Limits the dataset size to 500 samples.
        ensemble_size (int): Number of models per ensemble.
        scenario_name (str): Name of the run. Used for naming of save files.
        max_run (int): Number of repetitions of the scenario.
        recalibration (bool): If true, the models are recalibrated and evaluated again.
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

        metrics_eval_df = train_and_eval_nll_ensemble(train=train,
                                                    test=test,
                                                    yfeature=yfeature,
                                                    random_state=random_state,
                                                    dname=dname,
                                                    ensemble_size = ensemble_size,
                                                    recalibration=recalibration)
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
    parser.add_argument("--recalibration", action="store_true", help="Shall the models be calibrated after training?")
    args = parser.parse_args()
    logging.info('Setting log level to: %i ', int(args.log_level))
    logging.getLogger().setLevel(int(args.log_level))

    ensemble_size = 5
    scenario_name = args.name
    n_runs = args.n_runs

    if args.recalibration:
        recalibration = True
    else:
        recalibration = False



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
                            ensemble_size = ensemble_size,
                            scenario_name= 'sythetic',
                            max_run = n_runs,
                            recalibration=recalibration)
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
                            ensemble_size = ensemble_size,
                            scenario_name= 'real',
                            max_run = n_runs,
                            recalibration=recalibration)
            logging.info('Finished running real world datasets...')

        # Concatenate results.
        concat_excel_files(result_dir, keyword = 'metrics_', keep_originals=True)