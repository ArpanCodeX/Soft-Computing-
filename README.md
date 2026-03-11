import numpy as np
import matplotlib.pyplot as plt

# Sample Dataset
X = np.array([[1,2],
              [2,3],
              [3,5],
              [5,7],
              [3,1],
              [2,1]])

y = np.array([1,1,1,1,-1,-1])

# Initialize parameters
learning_rate = 0.01
epochs = 20

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

cost_list = []

# Training ADALINE
for epoch in range(epochs):

    net_input = np.dot(X, w) + b
    output = net_input
    
    error = y - output
    
    w = w + learning_rate * np.dot(X.T, error)
    b = b + learning_rate * error.sum()
    
    cost = (error**2).sum()/2
    cost_list.append(cost)

print("Final Weights:", w)
print("Final Bias:", b)

# Prediction
net_input = np.dot(X, w) + b
prediction = np.where(net_input >= 0, 1, -1)

print("Predictions:", prediction)

# Plot training error
plt.plot(range(1, epochs+1), cost_list)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("ADALINE Error Reduction")
plt.show()










import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([-1,1,1,-1])

# parameters
lr = 0.01
epochs = 20

# initialize weights
w = np.zeros(X.shape[1])
b = 0

cost_list = []

for epoch in range(epochs):

    net_input = np.dot(X,w) + b
    output = net_input

    error = y - output

    w = w + lr * np.dot(X.T,error)
    b = b + lr * error.sum()

    cost = (error**2).sum()/2
    cost_list.append(cost)

print("Final Weights:",w)
print("Bias:",b)

# prediction
net_input = np.dot(X,w)+b
pred = np.where(net_input>=0,1,-1)

print("Actual:",y)
print("Predicted:",pred)

plt.plot(range(1,epochs+1),cost_list)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("ADALINE Training Error on XOR")
plt.show()
