import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility
import torch
from skorch import NeuralNetClassifier
import plotly.figure_factory as ff


st.set_page_config(page_title="Train", page_icon=":shark:", layout="wide")
st.title("Train Your Model")

for k in st.session_state:
    st.session_state[k] = st.session_state[k]

# add line after the title
st.markdown("---")

st.write("""
            When training a model there are many parameters that can affect the quality of the training and training period. For example if we keep the value of learning rate too high, 
            then it causes drastic updates, which may lead to divergent behavior. When the learning rate is too low it takes forever to train the model to achieve the minimum loss point.
         So it should always be kept in mind that learning rate should be set accorldingly. Similary we have parameters like number of epochs, loass functions used for training.

         You can train the selected model, by choosing the configuration from below
         """)


#  ADD a interactive slider to select the number of images to display using streamlit.slider()

st.write("### TRAIN PARAMETERS")

# if "epochs" not in st.session_state:
#     epochs = 20
# else:
#     epochs = st.session_state["epochs"]

epochs = st.slider(label = "NO. OF EPOCHS", min_value=1, max_value=100, value=20, step=1)

# if "learning_rate" not in st.session_state:
#     learning_rate = 0.1
# else:
#     learning_rate = st.session_state["learning_rate"]

learning_rate = st.slider(label = "LEARNING RATE", min_value=0.01, max_value=1.00, value=0.01, step=0.01)


if st.button("TRAIN MODEL"):
    st.write("created: ",st.session_state["created"])
    st.write("created: ",st.session_state["split_ratio"])
    st.write("created: ",st.session_state["total_samples"])
    if "split_ratio" not in st.session_state or  "total_samples" not in st.session_state or "created" not in st.session_state:
        st.error("First set the dataset from dataset page, and then create the model")
    else:
        if "features" not in st.session_state:  
            with st.spinner("Loading training data..."):
                mnist = utility.Load_mnistData()
    
                X = mnist.data.astype('float32')
                y = mnist.target.astype('int64')
                X = X/255.
                
                features = X.astype(np.float32)
                st.session_state["features"] = features
                target = y.astype(np.int64)
                st.session_state["target"] = target
                


        split_ratio = st.session_state["split_ratio"]
        totalLen = st.session_state["total_samples"]

        Xsim,Xcus = st.session_state["features"],st.session_state["features"]
        if st.session_state["simpleModelType"] == "CNN":
            Xsim = st.session_state["features"].reshape(-1, 1, 28, 28)
            print(Xsim.shape)
            print(st.session_state["target"].shape)

        if st.session_state["customModelType"] == "CNN":
            Xcus = st.session_state["features"].reshape(-1, 1, 28, 28)
            print(Xcus.shape)
            print(st.session_state["target"].shape)

        Xsim_train, Xsim_test, y_train, y_test = train_test_split(Xsim[:totalLen], y[:totalLen], train_size = split_ratio/100, random_state=42)
        Xcus_train, Xcus_test, y_train, y_test = train_test_split(Xcus[:totalLen], y[:totalLen], train_size = split_ratio/100, random_state=42)

        assert(len(Xsim_train)+len(Xsim_test) == totalLen)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        model = st.session_state["simpleModel"]
        customModel = st.session_state["customModel"]

        simnet = NeuralNetClassifier(model,max_epochs=epochs,lr=learning_rate,optimizer=torch.optim.Adam,device=device,callbacks=[('custom', utility.CustomCallback())])
        cusnet = NeuralNetClassifier(customModel,max_epochs=epochs,lr=learning_rate,optimizer=torch.optim.Adam,device=device,callbacks=[('custom', utility.CustomCallback())])

        st.write(Xsim_train.shape, y_train.shape)
        st.write(Xsim_test.shape, y_test.shape)

        with st.spinner("training model..."):
            simnet.fit(Xsim_train,y_train)
            st.write("st: ",st.session_state["hidden_layers"])
            cusnet.fit(Xcus_train,y_train)
            st.write("Models Trained Successfully")
            st.session_state["simnet"] = simnet
            st.session_state["cusnet"] = cusnet
            st.session_state["Xsim_test"] = Xsim_test
            st.session_state["Xcus_test"] = Xcus_test
            st.session_state["y_test"] = y_test
