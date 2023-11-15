import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility
from streamlit_drawable_canvas import st_canvas
import cv2
from sklearn.metrics import accuracy_score

# from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="Predict", page_icon=":shark:", layout="wide")
st.title("Make Predictions On Your Data")

for k in st.session_state:
    st.session_state[k] = st.session_state[k]

# add line after the title
st.markdown("---")

st.write("""
         Lets test how our model performs on test data""")

st.write("##")
st.write("""
         ## Test Data
        
         """)

if st.button("predict with data"):
    if "cusnet" not in st.session_state or "simnet" not in st.session_state:
        st.error("Train the models first before predicting")
    else:
        Xsim_test = st.session_state["Xsim_test"]
        Xcus_test = st.session_state["Xcus_test"]
        y_test = st.session_state["y_test"]
        simnet = st.session_state["simnet"]
        custnet = st.session_state["cusnet"]
        simpreds = simnet.predict(Xsim_test)
        cuspreds = custnet.predict(Xcus_test)
        st.write(f"accuracy of simple model is {accuracy_score(y_test,simpreds)}")
        st.write(f"accuracy of custom model is {accuracy_score(y_test,cuspreds)}")


#  ADD a interactive slider to select the number of images to display using streamlit.slider()

# samples = st.slider(label = "SAMPLE SIZE", min_value=10, max_value=100, value=20, step=10)

# if st.button("Visualize Data"):
#     if "net" not in st.session_state:
#         st.error("train a model first before trying to predict")
#     else:
#         x_test = st.session_state["X_test"]
#         y_test = st.session_state["y_test"]
#         utility.plot_example(x_test,y_test, samples)


custom_css = """
    <style>
        .stCanvasToolbar {
            background-color: #ffffff !important;
        }
        .st
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

canvas = st_canvas(
        key="canvas",
        width=600,
        height=400,
        background_color="#ffffff",
        drawing_mode="freedraw",  # other modes: "line", "rect", "circle", "transform"
        update_streamlit=True,
    )

if st.button("Predict"):
    # st.write(st.session_state["simnet"])
    # st.write(st.session_state["cusnet"])

    if "simnet" not in st.session_state or "cusnet" not in st.session_state:
        st.error("Train the models first before predicting")
    else:
        image = canvas.image_data  # Ensure the image is in RGB format
        cv2.imwrite('image.png', image)
        image_path = "image.png"
        image = Image.open(image_path)
        image = image.convert("L")
        image = image.resize((28, 28))
        X_test1, X_test2 = np.array(image), np.array(image)
        
        X_test1, X_test2 = cv2.bitwise_not(X_test1), cv2.bitwise_not(X_test2)
        X_test1, X_test2 = X_test1.astype(np.float32), X_test2.astype(np.float32)

        cv2.imwrite('image1_uint8.png', X_test1)
        cv2.imwrite('image2_uint8.png', X_test2)

        
        if st.session_state["simpleModelType"] == "ANN":
            X_test1 = X_test1.reshape(1, 784)
        elif st.session_state["simpleModelType"] == "CNN":
            X_test1 = X_test1.reshape(-1,1,28,28)

        if st.session_state["customModelType"] == "ANN":
            X_test2 = X_test2.reshape(1, 784)
        elif st.session_state["customModelType"] == "CNN":
            X_test2 = X_test2.reshape(-1,1,28,28)

        print(X_test1.shape)

        X_test1 /= 255.0
        X_test2 /= 255.0
        
        simnet = st.session_state["simnet"]
        custnet = st.session_state["cusnet"]
        simpreds = simnet.predict(X_test1)
        cuspreds = custnet.predict(X_test2)

        st.write("prediction from simple model is ", simpreds)
        st.write("prediction from custom model is ", cuspreds)

