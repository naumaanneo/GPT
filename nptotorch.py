import numpy as np
import torch 


# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])


t_data = torch.tensor(data)

t_allytrue = torch.tensor(all_y_trues)

print (t_data, t_data.shape)


y_preds = torch.stack([torch.tensor(x) for x in t_data], dim=1)
print(y_preds)