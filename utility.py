import matplotlib.pyplot as plt
import streamlit as st
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml


def plot_example(X, y, size):
    rows = int(size/10)
    fig = plt.figure(figsize=(15, 15))  

    for i in range(rows): 
        for j in range(10):
            index = i * 10 + j
            plt.subplot(10, 10, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            plt.xticks([])  # Remove x-ticks
            plt.yticks([])  # Remove y-ticks
            plt.title(y[index], fontsize=8)  # Display the label as title with reduced font size

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
    st.pyplot(fig)


def NerualNet(X, y):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(y))

    class ClassifierModule(nn.Module):
        def __init__(
                self,
                input_dim=mnist_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.5,
        ):
            super(ClassifierModule, self).__init__()
            self.dropout = nn.Dropout(dropout)

            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X
    
    
    torch.manual_seed(0)
    

    class CustomCallback(Callback):
        def on_epoch_end(self, net, **kwargs):
            epoch = net.history[-1]['epoch']
            train_loss = net.history[-1]['train_loss']
            valid_acc = net.history[-1]['valid_acc']
            valid_loss = net.history[-1]['valid_loss']
            duration = net.history[-1]['dur']
            # y_pred = net.predict(X_test)
            # model_acc = accuracy_score(y_test, y_pred)
            st.write(f"Epoch {epoch}: Train Loss {train_loss:.4f} Valid Accuracy = {valid_acc:.4f} Valid Loss = {valid_loss:.4f} Duration = {duration:.4f}.")
           
    
    net = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=20,
        lr=0.1,
        device=device,
        callbacks=[('custom', CustomCallback())],
    )

    return net

def train(net, learning_rate, epochs, X_train, y_train):
    net.set_params(max_epochs=epochs, lr=learning_rate)
    net.fit(X_train, y_train)
    return net


class CNNDepth(nn.Module):
    def __init__(self,in_channels,actv,depth_layer=10):
        super().__init__()
        self.in_channels = in_channels
        self.activation = actv
        self.depth = depth_layer
        self.conv_layers = self.create_layers(depth_layer)

        self.fcs = nn.Sequential(
            nn.Linear(49,20,bias=True),
            nn.Dropout(p=0.3),
            nn.Linear(20,10),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        return x

        pass

    def create_layers(self,depth):
        layers = []
        in_channels = self.in_channels

        actvationFunction = nn.ReLU()
        if self.activation == "Sigmoid":
            actvationFunction = nn.Sigmoid()
        elif self.activation == "Tanh":
            actvationFunction = nn.Tanh()


        
        for x in range(depth):
            layers += [
                nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64),
                actvationFunction
            ]
            in_channels = 64

        layers += [
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,32,3,1,1),
            actvationFunction,
            nn.Conv2d(32,16,3,1,1),
            actvationFunction,
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,1,1,1),
            nn.Dropout(p=0.1),
            ]

        return nn.Sequential(*layers)

class NeuralNetClass(nn.Module):
    # input_dims = 784 -> total 784 pixels
    # output_dims = 10 -> 10 unique classes
    def __init__(self,input_dim=784,hidden_dim=98,output_dim=10,dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(input_dim,hidden_dim)
        self.output_layer = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = F.softmax(self.output_layer(x),dim = -1)
        return x
    
class NeuralNetWithHiddenLayers(nn.Module):
    def __init__(self,input_dim=784,hidden_dims=[98],output_dim=10,dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.hidden_layers = nn.Linear(input_dim,hidden_dims[0])
        self.output_layers = nn.Linear(hidden_dims[0],output_dim)
        layers = []
        if len(hidden_dims) > 1:
            for i in range(1,len(hidden_dims)):
                layers += [
                    nn.Linear(hidden_dims[i-1],hidden_dims[i]),
                    nn.Dropout(dropout),
                    nn.ReLU()
                ]
            self.output_layers = nn.Linear(hidden_dims[-1],output_dim)
        self.interim_layers = nn.Sequential(*layers)

    def forward(self,x):
        x = self.hidden_layers(x)
        if len(self.hidden_dims)>1:
            x = self.interim_layers(x)
        x = F.softmax(self.output_layers(x),dim=-1)
        return x

            
class CNN(nn.Module):
    def __init__(self,dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600,100) # numer_channels*width*size
        self.fc2 = nn.Linear(100,10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self,x):
        x = torch.relu(F.max_pool2d(self.conv1(x),2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2)) 

        x = x.view(-1,x.size(1)*x.size(2)*x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x),dim = -1)
        return x


    
# model returns:

def CreateModelWithDepth(in_channels=1,actv="ReLU",depth_layer=10):
    model = CNNDepth(in_channels,actv,depth_layer)
    return model


def createNeuralNet(hidden_dims):
    model = NeuralNetClass(hidden_dim = hidden_dims)
    return model

def createModelWithHiddenLayers(hidden_dims,dropout=0.5):
    model = NeuralNetWithHiddenLayers(hidden_dims=hidden_dims,dropout=dropout)
    return model

def createCNN(dropout=0.5):
    model = CNN(dropout=dropout)
    return model

# TODO: Complete this function
# def loadModelToTrain()

@st.cache_data
def Load_mnistData():
    mnist = fetch_openml('mnist_784',as_frame=False,cache=True)
    return mnist

class CustomCallback(Callback):
    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1]['epoch']
        train_loss = net.history[-1]['train_loss']
        valid_acc = net.history[-1]['valid_acc']
        valid_loss = net.history[-1]['valid_loss']
        duration = net.history[-1]['dur']
        # y_pred = net.predict(X_test)
        # model_acc = accuracy_score(y_test, y_pred)
        st.write(f"Epoch {epoch}: Train Loss {train_loss:.4f} Valid Accuracy = {valid_acc:.4f} Valid Loss = {valid_loss:.4f} Duration = {duration:.4f}.")