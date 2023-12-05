#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import altair as alt

# Set title
st.title('TEAM 6: DEEP LEARNING FOR WILDFIRE DETECTION AND PREDICTION')

# Header
st.sidebar.markdown("---")
st.sidebar.markdown("Team 6 - Wildfire Dectection")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Topic", "Agenda", "Wildfire Definition", "Video of wildfire", "Fire Image", "No Fire Image", "Problem Statement and Objective", "Dataset Overview", "Workflow Diagram",'Models','Prediction', "Conclusion", "Project Summary"])

if page == "Topic":
    # Modelling Images
    st.header('TOPIC')
    topic_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide1.JPG'
    # Display image
    st.image(topic_url)

elif page == "Agenda":
    # Modelling Images
    st.header('AGENDA')
    agenda_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide2.JPG'
    # Display image
    st.image(agenda_url)

elif page == "Wildfire Definition":
    # Modelling Images
    st.header('WILDFIRE DEFINITION')
    agenda_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide3.JPG'
    # Display image
    st.image(agenda_url)

elif page == "Video of wildfire":
    # URL to MOV file
    video_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/fire1.mov'
    # Display video
    st.video(video_url)

elif page == "Fire Image":
    # Modelling Images
    st.header('FIRE IMAGE')
    fire_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide4.JPG'
    # Display image
    st.image(fire_url)

elif page == "No Fire Image":
    # Modelling Images
    st.header('NO FIRE IMAGE')
    nofire_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide5.JPG'
    # Display image
    st.image(nofire_url)

elif page == "Problem Statement and Objective":
    # Modelling Images
    st.header('PROBLEM STATEMENT AND OBJECTIVE')
    prob_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide6.JPG'
    # Display image
    st.image(prob_url)

elif page == "Dataset Overview":
    # Visualization graphs
    st.header('Visualization Graphs')

    # Dummy data for visualization
    data = {'Class': ['Fire', 'NoFire'], 'Count': [5000, 3600]}
    df = pd.DataFrame(data)

    # Bar graph for Fire
    st.subheader('Bar Graph for Fire and NoFire Images')
    fig_fire = alt.Chart(df).mark_bar().encode(
        x='Class',
        y='Count',
        tooltip=['Class', 'Count']
    ).interactive()
    st.altair_chart(fig_fire, use_container_width=True)

    # Fire images
    data1 = {'Class': ['Train', 'Test', 'Valid'], 'Count': [3600, 700, 700]}
    df1 = pd.DataFrame(data1)

    # Bar graph for Fire
    st.subheader('Bar Graph for Fire Images')
    fig_fire1 = alt.Chart(df1).mark_bar().encode(
        x='Class',
        y='Count',
        tooltip=['Class', 'Count']
    ).interactive()
    st.altair_chart(fig_fire1, use_container_width=True)

    # NoFire images
    data2 = {'Class': ['Train', 'Test', 'Valid'], 'Count': [3200, 200, 200]}
    df2 = pd.DataFrame(data2)
    # Bar graph for NoFire
    st.subheader('Bar Graph for NoFire Images')
    fig_fire2 = alt.Chart(df2).mark_bar().encode(
        x='Class',
        y='Count',
        tooltip=['Class', 'Count']
    ).interactive()
    st.altair_chart(fig_fire2, use_container_width=True)

elif page == "Workflow Diagram":
    # Modelling Images
    st.header('WORKFLOW DIAGRAM')
    work_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide8.JPG'
    # Display image
    st.image(work_url)

elif page == 'Models':
    st.header('Different models we tried out.')


    tab1, tab2 = st.tabs(["ResNet50", "VGG16"])

    tab1.subheader("ResNet50")
    tab1.write('First, we tried out the ResNet50 model to train the images and got the following outputs:')
    tab1.url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/resnet1.jpeg?raw=true'
    tab1.image(tab1.url)
    tab1.write('The above confusion matrix is for the ResNet Model')
    tab1.write('The next image is the accuracy of the model on the train and validation set')
    tab1.url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/resnet2.jpeg?raw=true'
    tab1.image(tab1.url)
    tab1.write("""
The accuracy for the Resnet model is quite high. But...

The next image sorts out why we chose the VGG instead of Resnet.
    """)
    tab1.url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/resnet3.jpeg?raw=true'
    tab1.image(tab1.url)
    tab1.write("""
Here, we can see that the loss on the validation set is quite high at 1.6.

This means that our model may be overfitting, but we saw that there is noise in the images where there is fire and smoke.
    """)


elif page == "Prediction":
    import streamlit as st
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.resnet import preprocess_input
    import tensorflow as tf
    from keras.models import load_model
    from PIL import Image
    import numpy as np

    # Load the model
    model = load_model('finalmodel/model.h5')


    def preprocess_image(img):

        img = img.resize((224, 224))


        if img.mode != 'RGB':
            img = img.convert('RGB')


        img_array = img_to_array(img)


        img_array = preprocess_input(img_array)


        img_array = np.expand_dims(img_array, axis=0)

        return img_array


    st.title("Wildfire Prediction")

    # Add a slider for the confidence threshold
    confidence_threshold = st.sidebar.slider("Set the confidence threshold (%)", 0, 100, 50)  # Start at 50%

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        class_names = ['fire', 'no fire']
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Check if the confidence is above the threshold
        if confidence >= confidence_threshold:
            predicted_class = class_names[class_index]
            st.write(f"Prediction: {predicted_class} (Class {class_index})")
            st.write(f"Confidence Score: {confidence:.2f}%")
        else:
            st.write("Confidence score too high to make a prediction.")

elif page == "Conclusion":
    # Modelling Images
    st.header('CONCLUSION')
    con_url = 'https://raw.githubusercontent.com/Archonz-crazy/DL_final_project/main/code/resources/Slide9.JPG'
    # Display image
    st.image(con_url)

elif page == "Project Summary":

    st.title("Team 6 AI - Wildfire Detection Project")

    st.write(
        """
        A wildfire, also known as a forest fire, bushfire, or vegetation fire, is an uncontrolled fire that spreads rapidly through vegetation, such as forests, grasslands, or shrublands.
        Wildfires can be ignited by various sources, including lightning, human activities, or volcanic eruptions. Our objective is to develop a deep learning-based system for early wildfire detection and prediction.
        We utilize over 10,000 images from various types of wildfires such as forest fires, grass fires, and bushfires.
        """
    )

    st.title("Implementation Plan")

    st.write(
        """
        1. Data Preparation:
           - Load and preprocess the FlameVision Dataset.
           - Augment training data for model robustness.

        2. Model Selection:
           - Choose Convolutional Neural Network as the deep learning architecture.
           - Explore pre-trained models and transfer learning techniques.

        3. Model Training:
           - Adjust hyperparameters.
           - Implement early stopping and learning rate scheduling.
           - Evaluate performance on the validation set.

        4. Model Evaluation:
           - Assess the model's performance on the test set.
           - Analyze strengths and weaknesses.

        5. Fine-tuning and Optimization:
           - Adjust the model based on performance analysis.
           - Experiment with optimization techniques (learning rates, regularization, dropout).

        6. Real-time Detection and Prediction:
           - Develop a pipeline for real-time wildfire images or video streams.
           - Apply the trained model for detection and prediction.
           - Implement post-processing to filter false positives.

        Finally, we trained a robust model that can accurately detect and classify different types of wildfires.
        The system utilizes deep learning algorithms and image processing techniques to analyze real-time wildfire images, enabling timely response and mitigation efforts.
        Thank you all for your attention!
        """
    )



