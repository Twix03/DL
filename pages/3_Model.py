import streamlit as st
import numpy as np
import pandas as pd
import utility
from skorch import NeuralNetClassifier


st.set_page_config(page_title="Model", page_icon=":shark:", layout="wide")
st.title("Build Your Own Model")

for k in st.session_state:
    st.session_state[k] = st.session_state[k]

# add line after the title
st.markdown("---")

st.write("""
        The second important parameter to decide is to choose which model/ training algorith to use for training. Each model has its unique strengths and limitations, 
         making them suitable for specific tasks. Understanding these strengths and limitations is crucial for selecting the most appropriate model for a given scenario.
    
        For example, Convolutional neural networks (CNNs) excel at image-related tasks, effectively capturing the intricate patterns and 
        features within visual data. Their ability to process spatial relationships makes them ideal for tasks such as image classification, 
        object detection, and image segmentation.
        
        In contrast for general classification purposes, where the input is a set of numerical features, artificial neural networks (ANNs) often emerge as the preferred choice. 
        Their versatility and ability to learn complex relationships between input features and output labels make them applicable to a wide range of problems.
        """)
# looks like this is not required: can remove these 

# if "samples" not in st.session_state:
#     samples = 50
# else:
#     samples = st.session_state["samples"]

# samples = st.slider(label = "NO. OF SAMPLES", min_value=10, max_value=100, value=20, step=10)


st.write("## Set the config of the models")

def simpleModels(modelType):
    if modelType == "ANN":
        return utility.createNeuralNet(98)
    else: return utility.createCNN(0.5)

with st.container():
    st.write("A simple model")
    selection = st.radio("Model",options=["ANN","CNN"],index=0,key="simpleModelType",horizontal=True)
    if st.button("create model"):
        simpleModel = simpleModels(selection)
        st.write("model created")
        if "simpleModelType" not in st.session_state:
            st.session_state["simpleModelType"] = selection
        st.session_state["simpleModel"]= simpleModel
        st.session_state["created"] = True

        st.write(st.session_state["simpleModel"])

# def presentCustomOptions(modelType):
#     st.write("you have selected to create a custom model")
#     if modelType == "ANN":
#         st.write("Selected custom ANN")
#         st.slider("dropout",0.,1.,value=0.5,step=0.1,key="dropout")
#         hiddenLayers = st.slider("no.of hidden layers",1,5,step=1,value=1)
#         st.write("below you can choose the number of nodes in each hidden layer: ")
#         sliderInputs = [0 for i in range(hiddenLayers)]
#         for i in range(hiddenLayers):
#             sliderInputs[i] = st.slider("choose numer of nodes",100,500,step=50,value=100)
#         st.session_state.sliderInputs = sliderInputs
#     else: 
#         st.write("Selected custome CNN")
#         st.write("**choose your function**")
#         st.radio("activation",options=['ReLU','Tanh','Sigmoid'],index=0,key="activation")        
#         st.write("**Choose the number of convolution layers in the model**")
#         st.slider("no.of layers",1,10,value=5,key="depth")

        


# modelType = st.radio("### Choose a model",options=["simple","custom"])



# # This code needs to be modified to function correctly
# pressed = st.button("show options")
# if pressed not in st.session_state:
#     st.session_state["pressed"] = True
# # This might be happening because the script is run after the selection
# if pressed:
#     with st.container():
#         model = None
#         if modelType == "simple":
#             with st.container():
#                 selection = st.radio("**select model** ",options=["ANN","CNN"],key="selectionSimple")
#                 # if selection == "ANN":
#                 #     model = utility.createNeuralNet(98)
#                 # if selection == "CNN":
#                 #     model = utility.createCNN(0.5)
#                 model = simpleModels(selection)
#                 st.write("Model Created Successfully")
#         else:
#             with st.container():
#                 selection = st.radio("**select model** ",options=["ANN","CNN"],key="selectionCustom")
#                 presentCustomOptions(selection)
#                 if selection == "ANN":
#                         model = utility.createModelWithHiddenLayers(st.session_state.sliderInputs,st.session_state["dropout"])
#                 if selection == "CNN":
#                     model = utility.CreateModelWithDepth(1,st.session_state['activation'],st.session_state['depth'])
#                 st.write("Model Created Successfully")
#         st.session_state["model"] = model

#     # features = st.session_state["features"]
#     # target = st.session_state["target"]
    
#     # net = utility.NerualNet(features, target)
    
#     # if "net" not in st.session_state:
#     #     st.session_state["net"] = net
#     # else:
#     #     net = st.session_state["net"]

#     # net = utility.CreateModelWithDepth(1,actv=actv,depth_layer=depth)

#     # if "net" not in st.session_state:
#     #     st.session_state.net = net

