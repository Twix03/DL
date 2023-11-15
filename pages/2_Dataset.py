import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility


st.set_page_config(page_title="Datasets", page_icon=":shark:", layout="wide")
st.title("Create Your Own Dataset")

for k in st.session_state:
    st.session_state[k] = st.session_state[k]

# add line after the title
st.markdown("---")

st.write("""
         ## MNIST Dataset
         For this excercise we will be training a model to learn to identify handwritten digits. In thr previous section it is mentioned that
         one of the important thing for training a model is to have a proper dataset. This is where **MNIST** dataset comes into picture.
         

        The MNIST dataset comprises a vast collection of handwritten digits, encompassing a wide range of writing styles and variations, this dataset also 
        serves as a benchmark for evaluating and comparing various algorithms.
         
        The MNIST dataset consists of a total of 70,000 images. Each image is a handwritten digit, labeled with its corresponding value (0 to 9). 
        The images are grayscale and have been normalized to a uniform size of 28x28 pixels.
         
        The MNIST dataset's simplicity and well-structured format make it an ideal choice for beginners to explore the fundamentals of machine learning.
        We will see how the size of the traning data will affect the performance of the model, when all the other parameters are kept same
         """)

st.write("##")
st.write("To Visualize the data present in the dataset. Use the slider below to select the sample size to visualize and click on visulaize")

#  ADD a interactive slider to select the number of images to display using streamlit.slider()
# TODO: I Think this can be removed, not required
# if "samples" not in st.session_state:
#     samples = 50
# else:
#     samples = st.session_state["samples"]

samples = st.slider(label = "SAMPLE SIZE", min_value=10, max_value=100, value=20, step=10)

if st.button("Visualize Data"):
    mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255.0
    # file_features = "mnist_features.npy"
    # file_target = "mnist_target.npy"
    if "features" not in st.session_state:
        features = X.astype(np.float32)
        st.session_state["features"] = features

    else:
        features = st.session_state["features"]
    
    if "target" not in st.session_state:
        target = y
        st.session_state["target"] = target

    else:
        target = st.session_state["target"]
    
    

    utility.plot_example(features, target, samples)



# adjust width of slider
st.write("##")

st.write("""
         ## Create your own dataset: 
        
        you can select the total number of samples, and how many to choose for testing.
         
        The data is first scaled down to the range [0,1]
        """)

st.write("##")

total_samples = st.slider("no.of Images",10000,60000,5000,500,key="total_samples")
split_ratio = st.slider("train data percentage",10,100,70,5,key="split_ratio")

if "total_samples" not in st.session_state:
    st.session_state["total_samples"] = total_samples

if "split_ratio" not in st.session_state:
    st.session_state["split_ratio"] = split_ratio
