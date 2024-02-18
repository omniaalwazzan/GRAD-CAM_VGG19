import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.nn import Linear, Dropout
from sklearn.model_selection import train_test_split
#%%
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torchsummary import summary
path = r"C:\Users\Omnia\Desktop\Phd\DNA_methy\mVal_cv_feat.csv" 
DNA_df = pd.read_csv(path)

#%%

X = DNA_df.iloc[:, 1:-1]
y = DNA_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#%%
class TabularModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularModel, self).__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.dropout = Dropout(p=0.25)

    def forward(self, x):
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return (pred_y == y).sum().item() / len(y)

def train(model, X, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    epochs = 200

    for epoch in range(epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        acc = accuracy(out.argmax(dim=1), y)
        loss.backward()
        optimizer.step()

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:.2f}%')

    return model

def test(model, X, y):
    """Evaluate the model on the test set and print the accuracy score."""
    model.eval()
    out = model(X)
    acc = accuracy(out.argmax(dim=1), y)
    return acc
#%%
# Instantiate the tabular model
tabular_model = TabularModel(input_dim=X_train.shape[1], hidden_dim=400, output_dim=len(y_train.unique()))

# Train the model
train(tabular_model, torch.Tensor(X_train.values), torch.LongTensor(y_train.values))

# Test the model
acc = test(tabular_model, torch.Tensor(X_test.values), torch.LongTensor(y_test.values))
print(f'Tabular model test accuracy: {acc*100:.2f}%\n')

#%%
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)  # Convert to PyTorch tensor
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)  # Convert to PyTorch tensor

tabular_model.eval()
with torch.no_grad():
    outputs = tabular_model(X_test_tensor)
    probabilities = F.softmax(outputs, dim=1)  # Convert raw scores to probabilities
    _, predicted = torch.max(outputs, 1)      # Get the predicted class indices

y_true = y_test.values
y_score = probabilities.numpy()

# Compute accuracy
accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
#%%

num_classes = 20

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):  
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])  # Use binary labels for each class
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()


#%%

def integrated_gradients(model, input_tensor, baseline_tensor, num_steps=100):
    """
    Calculate integrated gradients for a given model and input tensor.
    
    Args:
        model (nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): The input tensor for which to compute the integrated gradients.
        baseline_tensor (torch.Tensor): The baseline tensor (often an input with all features set to zero).
        num_steps (int): The number of steps to use in the approximation.
    
    Returns:
        torch.Tensor: Integrated gradients for each feature.
    """
    delta = (input_tensor - baseline_tensor) / num_steps
    integrated_grads = torch.zeros_like(input_tensor)
    for i in range(num_steps + 1):
        interpolated_input = baseline_tensor + i * delta
        interpolated_input.requires_grad = True
        output = model(interpolated_input)
        grad = torch.autograd.grad(output.sum(), interpolated_input)[0]
        integrated_grads += grad
    integrated_grads *= delta
    return integrated_grads
#%%

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32) 
integrated_gradients_list = []
for sample in X_test_tensor:
    baseline_tensor = torch.zeros_like(sample)  # Baseline tensor for each sample
    integrated_grads = integrated_gradients(tabular_model, sample.unsqueeze(0), baseline_tensor.unsqueeze(0))
    integrated_gradients_list.append(integrated_grads)

integrated_gradients_tensor = torch.stack(integrated_gradients_list)

print("Shape of integrated gradients tensor:", integrated_gradients_tensor.shape) #torch.Size([306, 1, 8000])


#%%
integrated_gradients_array = integrated_gradients_tensor.detach().numpy().squeeze()
normalized_integrated_gradients = integrated_gradients_array / np.max(np.abs(integrated_gradients_array), axis=0)
mean_integrated_gradients = np.mean(normalized_integrated_gradients, axis=0)
num_features = len(mean_integrated_gradients)
plt.figure(figsize=(10, 6))
plt.bar(range(num_features), mean_integrated_gradients)
plt.xlabel('Feature Index')
plt.ylabel('Normalized Integrated Gradient')
plt.title('Feature Importance (Integrated Gradients)')
plt.show()

#%%

with torch.no_grad():
    output = tabular_model(X_test_tensor)
_, predicted_classes = torch.max(output, dim=1)
# this is a tensor of shape [num_samples] containing the predicted class indices
classes_tensor = predicted_classes

#%%
#### METHOD 1 ####
# Reshape integrated_gradients_tensor to remove the singleton dimension
integrated_gradients_array = integrated_gradients_tensor.squeeze().numpy()
results = []
# Iterate over each sample
for sample_index in range(len(integrated_gradients_array)):
    # Extract integrated gradients for the current sample
    integrated_gradients = integrated_gradients_array[sample_index]    
    # Get the predicted class for the current sample 
    predicted_class = classes_tensor[sample_index]
    
    # Iterate over each feature and its corresponding integrated gradient
    for feature_index, gradient_value in enumerate(integrated_gradients):
        results.append({
            'Sample Index': sample_index,
            'Feature Index': feature_index,
            'Integrated Gradient': gradient_value,
            'Predicted Class': predicted_class
        })

results_df = pd.DataFrame(results)
#results_df.to_csv('integrated_gradients_results.csv', index=False)

#%%
#### METHOD 2 ####
results = []

for sample_index in range(len(integrated_gradients_array)):
    integrated_gradients = integrated_gradients_array[sample_index]
    predicted_class = classes_tensor[sample_index]    
    results.append({
        'Sample Index': sample_index,
        'Predicted Class': predicted_class,
        **{f'Feature_{feature_index}': gradient_value for feature_index, gradient_value in enumerate(integrated_gradients)}
    })

results_df = pd.DataFrame(results)
#results_df.to_csv('integrated_gradients_results.csv', index=False)
#%%

#### METHOD 3 ####

results = []

for sample_index in range(len(integrated_gradients_array)):
    integrated_gradients = integrated_gradients_array[sample_index]
    
    predicted_class = classes_tensor[sample_index]
    # Get the original feature names from X_test
    feature_names = X_test.columns.tolist()
    # Append the results for the current sample
    results.append({
        'Sample Index': sample_index,
        'Predicted Class': predicted_class,
        **{feature_names[feature_index]: gradient_value for feature_index, gradient_value in enumerate(integrated_gradients)}
    })

results_df = pd.DataFrame(results)
