import numpy as np
import torch
import torch.nn as nn 



def sigmoid(data):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    sig = torch.nn.Sigmoid()
    sig1= sig(data)
 #   print (f" SIG OUT for {data} is {sig1}")
    return sig1

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    ds= sigmoid(x)
    d = ds * (1-ds)
  #  print (f" DER SIG OUT for {x} is {d }")
    return d 

def mse_loss(ytrue, ypred):
    mse = torch.mean(torch.square(ytrue - ypred))
   # print(f"MSE : {mse}")
    return mse

class OurNNTrain:

    def __init__(self):
        self.w1 = torch.randn(1)
        self.w2 = torch.randn(1)
        self.w3 = torch.randn(1)
        self.w4 = torch.randn(1)
        self.w5 = torch.randn(1)
        self.w6 = torch.randn(1)
        
        self.b1 = torch.randn(1)
        self.b2 = torch.randn(1)
        self.b3 = torch.randn(1)


    def feedforward(self, x):
        h1= sigmoid(self.w1 * x[0]  + self.w2 * x[1]  + self.b1)
        h2= sigmoid(self.w3 * x[0]  + self.w4 * x[1]  + self.b2)
        o1= sigmoid(self.w5 *  h1   + self.w3  * h2   + self.b3)
        return o1

    def train(self, data, allytrue):
        learnrate = 0.01
        epochs=10000

        for epoch in range(epochs):
         #  print(f" ITERATION : {epoch}")
            for x, y_true in zip(data, allytrue):

                #Do feedforward 
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                #Derivatives 
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learnrate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learnrate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learnrate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learnrate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learnrate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learnrate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learnrate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learnrate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learnrate * d_L_d_ypred * d_ypred_d_b3

                #Loss ?
                # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    #y_preds = torch.stack(self.feedforward(x) for x in data)
                    y_preds = torch.stack([self.feedforward(torch.tensor(x)) for x in data])
                    #y_preds = torch.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = torch.tensor([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

 
all_y_trues = torch.tensor([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNNTrain()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" DEVICE USED :::: {device}")
network.train(data, all_y_trues)


# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M