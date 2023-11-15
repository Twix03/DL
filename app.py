import streamlit as st

st.set_page_config(page_title="Learn DL", page_icon=":shark:", layout="wide")

st.title("Learn Deep Learning")

# add line after the title
st.markdown("---")

st.write("## Machine Learning")

st.write("""
    Have you ever wondered how does a self-driving car like Tesla work or how does instagram recommend posts which you might like? There are many more 
         questions like how does chatGPT or Google Bard answer our questions. The answer lies in **Machine Learning**, a very powerful tool 
         which helps computers to learn from data and make predictions without any explicit programming.    
""")

st.write("""
    Imagine teaching a child to recognize different animals. You start by showing them pictures of the animals like cat, dog, lion or a sparrow
         pointing their distinguishing features. After seeing enough examples, the child will be able to identify the animals on their own. Similarly we provide computers with vast amount of data, and they learn to identify patterns and realtionship within that data. This allows 
         computers to perform tasks like image recognition.    
""")

st.write("""
         ## How Do Computers See and learn?
         
         Just like us humans, computers needs to be trained to understand the things around them. The training process involves feeding the computer
         a large amount of data, like images, videos, audio, or text. The computer analyzes this data and tries to derive any pattern or relationships 
         from within the data.


         Once the computer has learned these features, it can be used to classify new images of cats and dogs.
         For example, if you want to train a compter on a dataset of cat and dog images, then it will to identify the distinguishing features
         of each animal, like tha height, shape of ears, lenght, shape, tail structure. Once the computer has learned these features it can be used
         to classify images of cats and dogs.

         Now this trained computer will be able to recognize between a cat and dog, but if you want it to identify othe animals, like lin or tiger 
         which it has never seen in the training data. It will not know what type of animal is that, as it does not have the knowledge or experience to identify them.  

        This is why it is important to use as much data as possible when training a computer model. 
         The more data the model learns from, the better it will be able to identify new patterns and make accurate predictions.
         In addition to using a large amount of data, it is also important to use right data. This means that the data should be accurate, consistent, and free of errors.
         When the computer learns on improper data, its ability to correctly perform the tasks will decrease.
    
         Along with data, the other parameter which plays a very important role is the right training algorithm for the task at hand.
        There are many different types of machine learning algorithms, and each one is designed for a specific purpose. For example, some algorithms are better at classifying images, while others are better at predicting numerical values.
         """)

st.write("##")

st.image("How.png", use_column_width=True)

st.write("It can be concluded by saying that to get the best results, high-quality data must be paired with the right training algorithm.")