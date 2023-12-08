elif page == 'Models':
    st.header('Different models we tried out.')
    url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/Slide10.JPG?raw=true'
    st.image(url)

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

    tab2.subheader("VGG16")
    tab2.write('Next, we tried out the VGG16 model to train the images and got the following outputs:')
    tab2.url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/vgg1.jpg?raw=true'
    tab2.image(tab2.url)
    tab2.write('The above confusion matrix is for the VGG16 Model')
    tab2.write('The next image is the accuracy of the model on the train and validation set')
    tab2.url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/vgg2.jpg?raw=true'
    tab2.image(tab2.url)
    tab2.write("""
    The accuracy for the VGG16 model is quite high. And...
        """)
    tab2.url = 'https://github.com/Archonz-crazy/DL_final_project/blob/main/code/resources/vgg3.jpg?raw=true'
    tab2.image(tab2.url)
    tab2.write("""
    Here, we can see that the loss on the validation set is very low at ~0.224.

    This means that our model is predicting images with or without fire in it, quite well.
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