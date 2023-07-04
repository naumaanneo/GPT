#pytorch example

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



inputdata = ([1,2])
x = torch.tensor(inputdata)

print(x, x.shape, x.dtype)

#calculate Dot product
w = torch.tensor([3,3])
print(w)

#Loop through for dot product
xw=0
for X in range(len(x)):
    
    print (f"Inputs : Index --> {X} Value ---> {x[X]}")
    for W in range(len(w)-1):
        print (f"Weights  : Index --> {W} Value ---> {w[W]}")
        xw += x[X]* w[W]
        print(xw)



#Push through Bias
b = 0.01
z= xw + b

print(z)

#print(f"T1 {torch.dot(torch.tensor([1, 2]), torch.tensor([3, 3]))}")
#print(f"T2 {torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))}")


####### Activation functions Sigmoid
x=torch.tensor([2,3])
w=torch.tensor([0,1])

#Loop through for dot product
xw=0
for X in range(len(x)):
    
    print (f"Inputs : Index --> {X} Value ---> {x[X]}")
    for W in range(len(w)-1):
        print (f"Weights  : Index --> {W} Value ---> {w[W]}")
        xw += x[X]* w[W]
        print(xw)

dot=torch.dot(x,w)
print(f"DOT :: {dot}")

#Push through Bias
b = 4
z= dot + b
print(z)

act=torch.sigmoid(dot)
print(f"ACT :: {act}")


    