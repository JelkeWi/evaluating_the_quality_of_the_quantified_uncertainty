import torch
import torch.nn as nn


class EnsembleMemberNLL(nn.Module):
    """
    A single ensemble member model for uncertainty quantification in the NLL ensembles.

    This class defines the architecture of a neural network that serves as an individual member of an ensemble.
    Each ensemble member predicts both the mean and variance of the target distribution, allowing for uncertainty
    estimation in predictions and the use of the NLL as the loss function.

    Attributes:
    -----------
    hidden_layers : nn.ModuleList
        A list of fully connected layers (Linear layers) representing the hidden layers in the network.

    mean_layer : nn.Linear
        The output layer that predicts the mean of the target distribution.

    logvar_layer : nn.Linear
        The output layer that predicts the log-variance of the target distribution.

    relu : nn.ReLU
        Activation function applied to the outputs of each hidden layer.

    Methods:
    --------
    forward(x)
        Defines the forward pass of the network. Computes the mean and variance of the target distribution for the input `x`.
    """

    def __init__(self, number_inputs, neurons_hidden, number_outputs, num_hidden_layers):
        """
        Initializes the ensemble member network with the specified number of inputs, hidden neurons, output neurons,
        and hidden layers.

        Parameters:
        -----------
        number_inputs : int
            The number of input features.

        neurons_hidden : int
            The number of neurons in each hidden layer.

        number_outputs : int
            The number of output neurons (typically 1 for regression tasks).

        num_hidden_layers : int
            The number of hidden layers in the network.
        """
        super(EnsembleMemberNLL, self).__init__()

        self.hidden_layers = nn.ModuleList()
        # Add the first layer (input to first hidden layer)
        self.hidden_layers.append(nn.Linear(in_features=number_inputs, out_features=neurons_hidden))

        # Add the remaining hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(in_features=neurons_hidden, out_features=neurons_hidden))

        # Output layer
        self.mean_layer = nn.Linear(in_features=neurons_hidden, out_features=number_outputs)
        self.logvar_layer = nn.Linear(in_features=neurons_hidden, out_features=number_outputs)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            The input data, typically of shape (batch_size, number_inputs).

        Returns:
        --------
        mean : torch.Tensor
            The predicted mean of the target distribution.

        var : torch.Tensor
            The predicted variance of the target distribution, ensured to be positive by applying an exponential function to the log-variance.
        """
        # Pass input through each hidden layer followed by ReLU
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        # Ensure variance is positive
        var = torch.exp(logvar)

        return mean, var


class DeepEnsembleNLL(nn.Module):
    """
    A deep ensemble of neural networks for uncertainty quantification with the NLL as the loss function.

    This class combines multiple `EnsembleMember` models into an ensemble. It aggregates predictions from each
    ensemble member to compute the ensemble's mean prediction and the associated uncertainty (variance).

    Attributes:
    -----------
    models : nn.ModuleList
        A list of `EnsembleMember` models representing the ensemble.

    Methods:
    --------
    forward(x)
        Defines the forward pass through the ensemble. Computes the ensemble mean and variance for the input `x`.
    """

    def __init__(self, num_members, input_size, hidden_layer_size, output_layer_size, num_hidden_layers):
        """
        Initializes the ensemble with the specified number of ensemble members and model architecture.

        Parameters:
        -----------
        num_members : int
            The number of ensemble members (i.e., individual neural networks) to include in the ensemble.

        input_size : int
            The number of input features for each ensemble member.

        hidden_layer_size : int
            The number of neurons in each hidden layer of the ensemble members.

        output_layer_size : int
            The number of output neurons for each ensemble member (typically 1 for regression tasks).

        num_hidden_layers : int
            The number of hidden layers in each ensemble member.
        """
        super(DeepEnsembleNLL, self).__init__()
        self.models = nn.ModuleList([EnsembleMemberNLL(
            input_size, hidden_layer_size, output_layer_size, num_hidden_layers) for _ in range(num_members)])

    def forward(self, x):
        """
        Defines the forward pass through the ensemble.

        Parameters:
        -----------
        x : torch.Tensor
            The input data, typically of shape (batch_size, input_size).

        Returns:
        --------
        ensemble_means : torch.Tensor
            The mean predictions of the ensemble, computed by averaging the means predicted by each ensemble member.

        ensemble_vars : torch.Tensor
            The variance predictions of the ensemble, computed by aggregating the variances and means predicted by each ensemble member.
        """
        predictions = [model(x) for model in self.models]  # List of tuples (mean, variance)

        # Unpack mean and variance from each tuple
        means_predictions = [pred[0] for pred in predictions]
        variances_predictions = [pred[1] for pred in predictions]

        # Stack means and variances to create one three-dimensional tensor for means/variances
        means_predictions = torch.stack(means_predictions, dim=0)
        variances_predictions = torch.stack(variances_predictions, dim=0)

        # Compute ensemble mean and variance for each data point
        ensemble_means = torch.mean(means_predictions, dim=0)
        # Compute full ensemble variance - take variances of individual members as well as different means into account
        ensemble_vars = torch.mean(variances_predictions + means_predictions ** 2, dim=0) - ensemble_means ** 2

        return ensemble_means, ensemble_vars  # Ensemble by averaging predictions and computing variance
