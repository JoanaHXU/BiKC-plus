import torch
import torch.nn as nn



class Binary_Classification_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.0):
        super(Binary_Classification_MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
 
    def compute_loss(self, x, target):
        """
        Computes the binary cross-entropy loss. 
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            target: Target tensor of shape (batch_size, output_dim).
        Returns:

            loss: Computed binary cross-entropy loss.
        """
        logits = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(logits, target)
        return loss
    

    def predict(self, x):
        """
        Predicts the binary classification output.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            predictions: Binary predictions of shape (batch_size, output_dim).
        """
        logits = self.forward(x)
        predictions = torch.sigmoid(logits)
        binary_predictions = (predictions > 0.5).float()
        return binary_predictions

