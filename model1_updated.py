
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from dataset import SentimentDataset
from model import SentimentModel  # Assuming the model is defined in model.py
import torch.optim as optim

# Set the hyperparameter grid for search
param_grid = {
    'hidden_units': [128, 256],
    'dropout_rate': [0.3, 0.5],
    'learning_rate': [1e-4, 5e-5],
    'batch_size': [16, 32],
    'epochs': [10, 20]
}

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare data
X = data[['input_ids', 'attention_mask']]
y = data['label']  # Adjust label column name as needed

best_params = None
best_score = 0.0

# Function for training and evaluating model
def train_and_evaluate(train_loader, val_loader, hidden_units, dropout_rate, learning_rate, epochs):
    model = SentimentModel(hidden_units=hidden_units, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training loop here (code not included for brevity)
    # ...

    # Return evaluation metric (like F1 score or accuracy)
    return evaluation_score

# Perform Grid Search to find the best hyperparameters
for hidden_units in param_grid['hidden_units']:
    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    fold_scores = []
                    for train_idx, val_idx in kfold.split(X):
                        train_data = data.iloc[train_idx]
                        val_data = data.iloc[val_idx]

                        train_dataset = SentimentDataset(train_data)
                        val_dataset = SentimentDataset(val_data)

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                        score = train_and_evaluate(train_loader, val_loader, hidden_units, dropout_rate, learning_rate, epochs)
                        fold_scores.append(score)

                    avg_score = np.mean(fold_scores)
                    print(f"Params: HU={hidden_units}, DR={dropout_rate}, LR={learning_rate}, BS={batch_size}, EP={epochs} => Avg F1: {avg_score}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'hidden_units': hidden_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }

# Best parameters after grid search
print(f"Best params: {best_params}, Best score: {best_score}")
