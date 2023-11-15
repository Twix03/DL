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

st.write("Let create our custom model. We have two options, one to create an ANN and the other is a custom CNN")

# looks like this is not required: can remove these 

# if "samples" not in st.session_state:
#     samples = 50
# else:
#     samples = st.session_state["samples"]

# samples = st.slider(label = "NO. OF SAMPLES", min_value=10, max_value=100, value=20, step=10)


st.write("## Set the config of the models")

with st.container():
    st.write("## Set the config of ANN Model")
    dropout = st.slider("dropout",0.,1.,value=0.5,step=0.1,key="dropout")
    hiddenLayers = st.slider("no.of hidden layers",1,5,step=1,value=1)
    st.write("below you can choose the number of nodes in each hidden layer: ")
    sliderInputs = [0 for i in range(hiddenLayers)]
    for i in range(hiddenLayers):
        sliderInputs[i] = st.slider("choose numer of nodes",100,500,step=50,value=100,key="h"+str(i))
    st.session_state["hidden_layers"] = sliderInputs

    if "dropout" not in st.session_state:
        st.session_state["dropout"] = dropout
    if "hidden_layers" not in st.session_state:
        st.session_state["hidden_layers"] = sliderInputs

with st.container():
    st.write("## Set the config for CNN Model")
    st.write("** Choose your function **")
    activation = st.radio("Activation",options=['ReLU','Tanh','Sigmoid'],index=0,key="activation")        
    st.write("**Choose the number of convolution layers in the model**")
    layers = st.slider("no.of layers",1,10,value=5,key="depth")

    if "activation" not in st.session_state:
        st.session_state["activation"] = activation
    if "depth" not in st.session_state:
        st.session_state["depth"] = layers


def customModels(modelType):
    if modelType == "ANN":
        st.write("hiddenLayers:" ,st.session_state["hidden_layers"])
        return utility.createModelWithHiddenLayers(st.session_state["hidden_layers"],st.session_state["dropout"])
    else: return utility.CreateModelWithDepth(1,st.session_state["activation"],st.session_state["depth"])

with st.container():
    st.write("### create your model by changing config and then click on train model")
    customSelection = st.radio("select model",options=["ANN","CNN"],index=0,key="customModelType",horizontal=True)
    if st.button("create Model"):
        customModel = customModels(customSelection)
        if "customModelType" not in st.session_state:
            st.session_state["customModelType"] = customSelection
        st.session_state["customModel"]= customModel
        if not st.session_state["created"]: st.session_state["created"]
        st.write("Custom Model Created")
        st.write(st.session_state["customModel"])



# with st.container():
#     st.write("custom model")
#     customSelection = st.radio("Model",options=["ANN","CNN"],index=0,key="customModelType",horizontal=True)
#     if st.button("create model"):
#         if customSelection == "ANN":
#             st.write("Selected custom ANN")
#             dropout = st.slider("dropout",0.,1.,value=0.5,step=0.1,key="dropout")
#             hiddenLayers = st.slider("no.of hidden layers",1,5,step=1,value=1)
#             st.write("below you can choose the number of nodes in each hidden layer: ")
#             sliderInputs = [0 for i in range(hiddenLayers)]
#             for i in range(hiddenLayers):
#                 sliderInputs[i] = st.slider("choose numer of nodes",100,500,step=50,value=100)
#             st.session_state.sliderInputs = sliderInputs
#         elif customSelection == "CNN":
#             st.write("Selected custome CNN")
#             st.write("**choose your function**")
#             activation = st.radio("activation",options=['ReLU','Tanh','Sigmoid'],index=0,key="activation")        
#             st.write("**Choose the number of convolution layers in the model**")
#             layers = st.slider("no.of layers",1,10,value=5,key="depth")
            
#         if customSelection == "ANN":
#             customModel = utility.createModelWithHiddenLayers(hidden_dims=hiddenLayers,dropout=dropout)
#         else: customModel = utility.CreateModelWithDepth(1,actv=activation,depth_layer=layers)
#         st.write("model created")
#         st.session_state["customModel"] = customModel
#         if "customModelType" not in st.session_state:
#             st.session_state["customModelType"] = customModel
#             st.session_state["customType"] = customSelection 



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

