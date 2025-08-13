import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging
from utils.nll_ensemble_model import DeepEnsembleNLL
import torch.optim as optim
from utils import eval_metrics
from copy import deepcopy
import uncertainty_toolbox as uct
import netcal.metrics
from netcal.regression import VarianceScaling, GPNormal, IsotonicRegression, GPBeta
from netcal import cumulative_moments

def train_and_eval_nll_ensemble(train: pd.DataFrame, test: pd.DataFrame, yfeature: str, random_state: int, dname: str, ensemble_size: int, recalibration: bool
                                ) -> pd.DataFrame:
    """
    A function to train and evaluate an ensemble of models trainined with the negative log-likelihood loss function.

    Args:
        train (pd.DataFrame): Training data
        test (pd.DataFrame): Testing data
        yfeature (str): Name of the target feature
        random_state (int): Random state
        dname (str): Name of the dataset.
        ensemble_size (int): Number of members in the ensemble.
        recalibration (bool): shall recalibration be run?
    Returns:
        metrics (pd.Dataframe): Containing all analysed evaluation metrics [mse, nll, picp, cal_score, sharpness_score, crps, ence, cwc, ecpe, uce, nll_pytorch].
    """
    # format train data
    x_train = torch.tensor(train.drop(columns=[yfeature]).values, dtype=torch.float)
    y_train = torch.tensor(train[yfeature].values, dtype=torch.float)

    # format test data
    x_test = torch.tensor(test.drop(columns=[yfeature]).values, dtype=torch.float)
    y_test = torch.tensor(test[yfeature].values, dtype=torch.float)
    #----------------------------------------------------------------------------------------------
    # Model Training
    #----------------------------------------------------------------------------------------------
    logging.info(f'Start training...')
    # model training
    ensemble = DeepEnsembleNLL(num_members=ensemble_size, 
                                input_size= x_train.shape[1],
                                num_hidden_layers= 2,
                                hidden_layer_size= 64,
                                output_layer_size= 1)
    optimizer = optim.Adam(ensemble.parameters(), lr=1E-3)
    criterion = torch.nn.GaussianNLLLoss()
    trainer = EnsembleTrainerOptimizedNLL(ensemble_model=ensemble, loss_fn=criterion, optimizer_func=optimizer)

    trainer.train(x_train, y_train, x_test, y_test, epochs=200, batch_size=32, random_state= random_state)

    # Inference
    logging.info(f'Start evaluation...')
    ensemble.eval()
    with torch.no_grad():
        prediction_mean, prediction_var = ensemble(x_test)
    prediction_mean = prediction_mean.detach().numpy().squeeze()
    prediction_var = prediction_var.detach().numpy().squeeze()
    prediction_std = np.sqrt(prediction_var)
    true_mean = y_test.numpy()
    target_range = (torch.max(torch.stack([y_train.max(), y_test.max()])) - 
                    torch.min(torch.stack([y_train.min(), y_test.min()])))
    target_range = target_range.item()

    logging.info('Begin evaluation ensemble')
    metrics_eval = eval_ensemble(prediction_mean=prediction_mean,
                            prediction_std=prediction_std,
                            true_mean= true_mean,
                            target_range = target_range)
    metrics_eval['calibration'] = 'none'
    logging.info('Finished evaluation ensemble')
    #----------------------------------------------------------------------------------------------
    # Calibration
    #----------------------------------------------------------------------------------------------
    if recalibration:
        # Calibration by Isotonic Regression
        logging.info('Starting calibration by isotonic regression')
        isotonic = IsotonicRegression()
        isotonic.fit((prediction_mean, prediction_std), true_mean)
        t_isotonic, pdf_isotonic, cdf_isotonic = isotonic.transform((prediction_mean, prediction_std))
        calibrated_mean, calibrated_var = cumulative_moments(t_isotonic, cdf_isotonic)
        calibrated_std = np.sqrt(calibrated_var)
        logging.info('Evaluation...')
        metrics_isotonic = eval_ensemble(prediction_mean=calibrated_mean.squeeze(),
                                prediction_std=calibrated_std.squeeze(),
                                true_mean= true_mean,
                                target_range = target_range)
        metrics_isotonic['calibration'] = 'isotonic'
        logging.info('Finished calibration by isotonic regression')
        
        # Calibration by variance scaling
        logging.info('Starting calibration by variance scaling')
        varscaling = VarianceScaling()
        varscaling.fit((prediction_mean, prediction_std), true_mean)
        calibrated_std = varscaling.transform((prediction_mean, prediction_std))
        logging.info('Evaluation...')
        metrics_variance = eval_ensemble(prediction_mean=prediction_mean,
                                prediction_std=calibrated_std.squeeze(),
                                true_mean= true_mean,
                                target_range = target_range)
        metrics_variance['calibration'] = 'variance'
        logging.info('Finished calibration by variance scaling')
        
        # Calibration by GB-Beta
        logging.info('Starting calibration by GP-Beta')
        gpbeta = GPBeta()
        gpbeta.fit((prediction_mean, prediction_std), true_mean)
        t_gpbeta, pdf_gpbeta, cdf_gpbeta = gpbeta.transform((prediction_mean, prediction_std))
        calibrated_mean, calibrated_var = cumulative_moments(t_gpbeta, cdf_gpbeta)
        calibrated_std = np.sqrt(calibrated_var)
        logging.info('Evaluation...')
        metrics_gpbeta = eval_ensemble(prediction_mean=calibrated_mean.squeeze(),
                                    prediction_std=calibrated_std.squeeze(),
                                    true_mean= true_mean,
                                    target_range = target_range)
        metrics_gpbeta['calibration'] = 'gpbeta'
        logging.info('Finished calibration by GP-Beta')

        # Calibration by GB-Normal
        logging.info('Starting calibration by GP-Normal')
        gpnormal = GPNormal()
        gpnormal.fit((prediction_mean, prediction_std), true_mean)
        calibrated_std = gpnormal.transform((prediction_mean, prediction_std))
        logging.info('Evaluation...')
        metrics_gpnormal = eval_ensemble(prediction_mean=prediction_mean,
                                prediction_std=calibrated_std.squeeze(),
                                true_mean= true_mean,
                                target_range = target_range)
        metrics_gpnormal['calibration'] = 'gpnormal'
        logging.info('Finished calibration by GP-Normal')

        metrics = pd.DataFrame([metrics_eval, metrics_isotonic, metrics_variance, metrics_gpnormal, metrics_gpbeta])
    else:
        metrics = pd.Series(metrics_eval).to_frame().T

    return metrics

def eval_ensemble(prediction_mean: np.ndarray, prediction_std: np.ndarray, true_mean: np.ndarray, target_range: float) -> dict:
    metrics = {}

    # Calculate Accuracy Metrics:
    # Calculates: mae, rmse, mdae, marpd, r2, corr
    # mdae = median absolute error
    # mardp = mean absolute relative percentage deviation
    accuracy_metrics = uct.get_all_accuracy_metrics(y_pred=prediction_mean, y_true=true_mean, verbose=False)
    metrics = metrics | accuracy_metrics # merges the two dicts

    # Calculate Scoring Rule Metrics:
    # Calculates: nll, crps, check, interval
    # check: check score from Gneiting 2007
    # interval: interval score from Gneiting 2007
    scoring_metrics = uct.get_all_scoring_rule_metrics(y_pred=prediction_mean, y_std= prediction_std, y_true=true_mean, scaled=True, resolution=20, verbose= False)
    metrics = metrics | scoring_metrics

    # Calculate calibration metrics:
    # Calculates: rms_cal, ma_cal, mis_cal
    # Meaning more or less unclear. Documentation of the uncertainty toolbox is rather bad. 
    calibration_metrics = uct.get_all_average_calibration(y_pred=prediction_mean, y_std= prediction_std, y_true=true_mean, num_bins=10, verbose= False)
    metrics = metrics | calibration_metrics

    # Calculate bounds for approximately 95% prediction interval
    lower_bounds_pi, upper_bounds_pi = eval_metrics.calculate_prediction_interval(
        prediction_mean, prediction_std, confidence_level=0.95)


    picp = netcal.metrics.PICP().measure(X=(prediction_mean, prediction_std), y=true_mean, q=0.95).picp.squeeze() # type: ignore
    metrics['picp'] = picp
    metrics['cwc'] = eval_metrics.calculate_coverage_width_based_criterion(lower_bounds_pi= lower_bounds_pi, upper_bounds_pi= upper_bounds_pi, picp= picp, target_range=target_range)
    metrics['sharpness_std'] = uct.sharpness(y_std= prediction_std)
    # Calculate Calibration Score - Kuleshov
    confidence_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    empirical_frequencies, cal_score = eval_metrics.calculate_calibration_score(prediction_mean, prediction_std, true_mean, confidence_levels)
    metrics['cal_score'] = cal_score
    
    metrics['ence'] = netcal.metrics.ENCE().measure(X=(prediction_mean, prediction_std), y=true_mean)
    metrics['uce'] = netcal.metrics.UCE().measure(X=(prediction_mean, prediction_std), y=true_mean)
    metrics['ecpe'] = eval_metrics.calculate_expectation_coverage_probability_error(mean_values=prediction_mean, std_values= prediction_std, true_values=true_mean, confidence_levels=confidence_levels)
    metrics['qce'] = netcal.metrics.QCE().measure(X=(prediction_mean, prediction_std), y=true_mean, q=np.linspace(0.1, 0.9, 9), reduction="mean")
    metrics['pinball'] = netcal.metrics.PinballLoss().measure(X=(prediction_mean, prediction_std), y=true_mean, q=np.linspace(0.1, 0.9, 9), reduction="mean")
    return metrics



class EnsembleTrainerOptimizedNLL:
    """
    A class to train an ensemble of models with bootstrap sampling, early stopping, and batching.
    This is the trainer used for the optimized NLL ensemble in main/run_ensemble_nll.py.

    This trainer initializes each model in the ensemble with a bootstrap sample of the training data
    and trains the model using the specified loss function and optimizer. Early stopping is used to
    halt training when validation loss no longer improves.

    Attributes:
    -----------
    ensemble : nn.Module
        The ensemble model consisting of multiple individual models.

    loss_fn : callable
        The loss function used to compute the loss during training.

    optimizer : torch.optim.Optimizer
        The optimizer used for training the models.

    train_losses : list
        List to store training losses for each model in the ensemble.

    val_losses : list
        List to store validation losses for each model in the ensemble.
    """

    def __init__(self, ensemble_model, loss_fn, optimizer_func):
        """
        Initializes the EnsembleTrainerOptimizedNLL with the ensemble model, loss function, and optimizer.

        Parameters:
        -----------
        ensemble_model : nn.Module
            The ensemble model to be trained.

        loss_fn : callable
            The loss function used for training.

        optimizer_func : torch.optim.Optimizer
            The optimizer used for training.
        """
        self.ensemble = ensemble_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_func
        self.train_losses = []  # To store training losses
        self.val_losses = []  # To store validation losses

    def train(self, training_input: torch.Tensor, training_output: torch.Tensor, val_input: torch.Tensor, val_output: torch.Tensor, epochs:int, batch_size:int, random_state):
        """
        Trains each model in the optimized NLL ensemble using bootstrap sampling, batching and early stopping.

        Parameters:
        -----------
        training_input : torch.Tensor
            The input features for training.

        training_output : torch.Tensor
            The output targets for training.

        val_input : torch.Tensor
            The input features for validation.

        val_output : torch.Tensor
            The output targets for validation.

        epochs : int
            The number of epochs to train each model.

        batch_size : int
            The batch size for training and validation data loaders.

        This method performs the following steps:
        1. For each model in the ensemble, a bootstrap sample of the training data is created.
        2. The model is trained using the bootstrap sample with early stopping based on the validation loss.
        3. Training and validation losses are recorded.
        4. After training, the best model weights (based on validation loss) are restored.
        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        # Train each model in the ensemble
        for idx, model in enumerate(self.ensemble.models):
            model.train()
            best_model = deepcopy(model.state_dict())
            early_stopping = EarlyStopping(patience = 10, min_delta = 0.0)
            # Create a bootstrap sample for this model
            bootstrap_indices = np.random.choice(len(training_input), size=len(training_input), replace=True)
            X_bootstrap = training_input[bootstrap_indices]
            y_bootstrap = training_output[bootstrap_indices]

            train_dataset = TensorDataset(X_bootstrap, y_bootstrap)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = TensorDataset(val_input, val_output)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model_train_losses = []
            model_val_losses = []
            best_val_loss = np.inf
            for epoch in range(epochs):
                epoch_train_loss = 0.0

                model.train()  # Ensure the model is in training mode
                for inputs, targets in train_loader:
                    self.optimizer.zero_grad()
                    pred_mean, pred_var = model(inputs)
                    loss = self.loss_fn(pred_mean, targets, pred_var)
                    loss.backward()
                    self.optimizer.step()

                    # Add up batch loss
                    epoch_train_loss += loss.item() * inputs.size(0)

                # Calculate average training loss for the epoch
                epoch_train_loss /= len(train_dataset)

                # Calculate validation loss
                epoch_val_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in val_loader:
                        val_pred_mean, val_pred_var = model(inputs)
                        val_loss = self.loss_fn(val_pred_mean, targets, val_pred_var)

                        epoch_val_loss += val_loss.item() * inputs.size(0)  # Accumulate batch loss

                    epoch_val_loss /= len(val_dataset)

                model_train_losses.append(epoch_train_loss)
                model_val_losses.append(epoch_val_loss)
                if epoch_val_loss < best_val_loss:
                    best_model = deepcopy(model.state_dict())
                    best_val_loss = epoch_val_loss
                early_stopping(epoch_val_loss)
                if early_stopping.early_stop:
                    logging.info('Early Stopping: Epoch ({}/{})'.format(epoch, epochs))
                    break
            model.load_state_dict(best_model)

            self.train_losses.append(model_train_losses)
            self.val_losses.append(model_val_losses)



class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            logging.debug(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logging.debug('INFO: Early stopping')
                self.early_stop = True