import torch
import torch.nn as nn


class Neuron:
    def __init__(self,weights,bias):
        self.weights=torch.tensor(weights,dtype=float)
        self.bias = bias

    def feedforward(self,inputs):
        inputs.to(float)
        dottotal=torch.dot(self.weights, inputs) + self.bias
        print(f" dottotal ={dottotal}")
        sig=torch.sigmoid(dottotal)
        print(f" sig ={sig}")
        return sig
    

# Feeding the neuron
weights = torch.tensor([0,1],dtype=float)
inputs = torch.tensor([2,3],dtype=float)
bias = 0
ex=Neuron(weights,bias)
func=ex.feedforward(inputs)
print(func)
 
layer1input=torch.tensor([func,func])
print(f" Layer 1 Input : {layer1input}")
layer1Neuron=Neuron(weights,bias=0)
layer1ff=layer1Neuron.feedforward(layer1input)
print(f" Layer 1 Output : {layer1ff}")


class OurNN:
    def __init__(self):
        weights= torch.tensor([0,1],dtype=float)
        bias = 0

        self.h1= Neuron(weights,bias)
        self.h2= Neuron(weights,bias)
        self.o1= Neuron(weights,bias)


    def feedforward(self, inp):

        #h1 takes inputs from inp 
        outh1=self.h1.feedforward(inp) 
        print(outh1)
        ##h2 takes input from inp
        outh2=self.h2.feedforward(inp)

        #o1 taek inputs from h1/h2 
        outo1= self.o1.feedforward(torch.tensor([outh1,outh2]))
 
        return outo1
    
print("Starting OurNN Simple! ")
ExOurNN = OurNN()
out= ExOurNN.feedforward(torch.tensor([2,3],dtype=float))
print(f" OUT from OurNN {out}")



### loss function : MSE Loss

def mse_loss(ytrue, ypred):
    mse = torch.mean(torch.square(ytrue - ypred))
    print("MSE : {mse}")
    return mse

yt=torch.tensor([1,0,0,1],dtype=float)
yp=torch.tensor([0,0,0,0],dtype=float)
p = mse_loss(yt,yp)
m = torch.nn.MSELoss()
loss=m(yt,yp)
print (f" MSE Derived LOSS : {p} -- > {loss}")