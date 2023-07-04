import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests

# Define the API endpoint and parameters
api_endpoint = "https://api.coingecko.com/api/v3"
coin_id = "bitcoin"
start_date = "1540995200"
end_date = "1703980800"

# Make the API request to fetch the Bitcoin price data
url = f"{api_endpoint}/coins/{coin_id}/market_chart/range"
params = {"vs_currency": "usd", "from": start_date, "to": end_date}
response = requests.get(url, params=params)
data = response.json()

# Extract the daily Bitcoin prices
prices = np.array(data["prices"])
print ( f" price {prices}" )
# Reverse the order to get the oldest prices first
prices = prices[:, 1][::-1]

# Create a copy of the numpy array to avoid negative stride issue
prices_copy = prices.copy()

# Set the random seed for reproducibility
torch.manual_seed(42)

# Convert the data to PyTorch tensors
inputs = torch.tensor(prices_copy[:-1], dtype=torch.float32)
labels = torch.tensor(prices_copy[1:], dtype=torch.float32)

 
# Normalize the data
mean = torch.mean(inputs)
std = torch.std(inputs)
inputs = (inputs - mean) / std

# Define the neural network
class BitcoinPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BitcoinPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return out

# Define the model and optimizer
input_size = 1  # Number of input features
hidden_size = 1000  # Number of units in the hidden layer
model = BitcoinPricePredictor(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model
num_epochs = 50000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs.unsqueeze(1))
    loss = criterion(outputs.squeeze(), labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Predict Bitcoin price by the end of 2023
input_prediction = torch.tensor(prices_copy[-1:], dtype=torch.float32)
prediction = []
model.eval()
with torch.no_grad():
    for _ in range(3650):  # Predicting for 365 days in 2023
        input_norm = (input_prediction - mean) / std
        output = model(input_norm.unsqueeze(0).unsqueeze(0))
        prediction.append(output.item())
        input_prediction = torch.tensor([output.item()], dtype=torch.float32)

# Denormalize the prediction
prediction = np.array(prediction) * std.item() + mean.item()

# Print the predicted Bitcoin price for the end of 2023
print("Predicted Bitcoin Price by the end of 2023:")
print(f"${prediction[-1]:.2f}")
