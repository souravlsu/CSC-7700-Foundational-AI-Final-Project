import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetFeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, criterion='MSE', activation_type='relu', regularization=None):
        super(ResNetFeedForwardNN, self).__init__()

        # Initializing input, output, and hidden dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.regularization = regularization

        # Choosing activation function
        self.activation = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }[activation_type.lower()]

        # Creating input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])

        # Creating hidden layers using ModuleList
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dim)):
            self.hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        # Creating output layer
        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

        # Adding optional dropout regularization
        if self.regularization == 'dropout':
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = None

        # Detecting device and moving model to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """
        Defining the forward pass including applying activations and optional dropout,
        and adding residual connections where dimensions match.
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = x          # Storing input for residual connection
            x = self.activation(layer(x))
            if self.dropout:
                x = self.dropout(x)
            if x.shape == residual.shape:
                x += residual     # Adding residual connection
        x = self.output_layer(x)  # Computing output
        return x

    def train_model(self, model, data_loader, optimizer):
        """
        Training the model using batches from the data loader.
        Computing MSE loss and updating weights.
        """
        model.train()
        running_loss = 0.0
        for batch in data_loader:
            x = batch[:, :model.input_dim].to(model.device)  # Splitting input features
            y = batch[:, model.input_dim:].to(model.device)  # Splitting target labels
            optimizer.zero_grad()                            # Resetting gradients
            y_pred = model(x)                                # Forward pass
            loss = torch.mean((y_pred - y) ** 2)             # Computing MSE loss
            loss.backward()                                  # Backpropagating error
            optimizer.step()                                 # Updating model parameters
            running_loss += loss.item()                      # Accumulating loss
        return model, running_loss / len(data_loader)

    def eval_model(self, model, data_loader):
        """
        Evaluating model performance on a validation set.
        Computing and returning the average loss.
        """
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                x = batch[:, :model.input_dim].to(model.device)
                y = batch[:, model.input_dim:].to(model.device)
                y_pred = model(x)
                loss = torch.mean((y_pred - y) ** 2).item()  # Computing loss
        return loss

    def test_model(self, model, data_loader):
        """
        Testing the model and returning all actual and predicted values.
        Collecting results across batches without updating the model.
        """
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[:, :model.input_dim].to(model.device)
                y = batch[:, model.input_dim:].to(model.device)
                y_pred = model(x)
                all_preds.append(y_pred.cpu())            # Collecting predictions
                all_targets.append(y.cpu())               # Collecting true values
        predicted_output = torch.cat(all_preds, dim=0)    # Concatenating predictions
        actual_output = torch.cat(all_targets, dim=0)     # Concatenating targets
        return actual_output, predicted_output
