import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageFile
from keras import activations, Model
from keras.src.applications import VGG16, ResNet50, NASNetLarge
from keras.src.layers import BatchNormalization, GlobalAveragePooling2D

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cwd = os.getcwd()
curr_path = os.path.dirname(cwd)
classification_path = os.path.join(curr_path, 'Classification')
train_dir = os.path.join(classification_path, 'train')
valid_dir = os.path.join(classification_path, 'valid')
test_dir = os.path.join(classification_path, 'test')
# Specify the filenames
weights_filename = 'yolov3.weights'
config_filename = 'yolov3.cfg'

# Create the full paths
weights_path = os.path.join(curr_path, weights_filename)
config_path = os.path.join(curr_path, config_filename)


def make_excel():
    # Getting the current working directory
    cwd = os.getcwd()

    # Navigate up to the 'Classification' directory
    curr_path = os.path.dirname(cwd)
    classification_path = os.path.join(curr_path, 'Classification')

    # Creating excel folder
    excel_folder = os.path.join(classification_path, 'excel')
    os.makedirs(excel_folder, exist_ok=True)

    # Excel file path
    excel_path = os.path.join(excel_folder, 'image_dataset_info.xlsx')

    # Check if Excel file already exists
    if not os.path.exists(excel_path):
        # DataFrame to hold the information
        data = {'ID': [], 'Split': [], 'Target': []}

        # Iterating through each split and class
        for split in ['train', 'test', 'valid']:
            for target in ['fire', 'nofire']:
                folder_path = os.path.join(classification_path, split, target)
                for image in os.listdir(folder_path):
                    data['ID'].append(image)
                    data['Split'].append(split)
                    data['Target'].append(target)

        # Creating a DataFrame
        df = pd.DataFrame(data)

        # Saving to Excel
        df.to_excel(excel_path, index=False)
        print("Excel file created at:", excel_path)
    else:
        print("Excel file already exists at:", excel_path)

    return excel_path


def data_vis():
    # Visualization of distribution by Target
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Target', data=data)
    plt.title('Distribution of Images by Target from train, test and valid')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 6))
    train_df = data[data['Split'] == 'train']
    sns.countplot(x='Target', data=train_df)
    plt.title('Distribution of Images by Target from train')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.show()

    # Visualization of distribution by Split
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Split', data=data)
    plt.title('Distribution of Images by Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.show()

    # Stacked Bar Chart
    pivot_df = data.groupby(['Split', 'Target']).size().unstack()
    pivot_df.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title('Distribution of Images by Target and Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.show()


def data_cleaning():
    # 1. Check for Missing Values
    print("Missing values before cleaning:")
    print(data.isnull().sum())

    # Drop rows with any missing values
    data.dropna(inplace=True)

    # 2. Removing Duplicates
    data.drop_duplicates(inplace=True)

    # 3. Data Type Conversion
    # Convert 'ID' and 'Target' to string if they're not already
    data['ID'] = data['ID'].astype(str)
    data['Target'] = data['Target'].astype(str)

    # 4. Text Data Cleaning
    # Example: Ensuring all 'Target' entries are lowercase
    data['Target'] = data['Target'].str.lower()

    # 5. Data Consistency
    # Example: Standardize category names
    data['Target'].replace({'fire': 'Fire', 'nofire': 'NoFire'}, inplace=True)

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    print("\nCleaned DataFrame:")
    print(data.head())


def preprocess_image(image_path):
    """
    Preprocess the image: Resize to 250x250 and ensure it has 3 channels.
    """
    with Image.open(image_path) as img:
        # Resize image and convert to RGB (3 channels)
        img = img.resize((250, 250)).convert('RGB')
    return img


def display_images(folder_path, title, num_examples=3):
    """
    Display a few example images from a specified folder.
    """
    images = os.listdir(folder_path)[:num_examples]
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))
    fig.suptitle(title)
    for ax, image in zip(axes, images):
        img_path = os.path.join(folder_path, image)
        img = preprocess_image(img_path)
        ax.imshow(img)
        ax.axis('off')
    plt.show()

'''
def save_augmented_images(class_name, num_images=1):
    """
    Save a specified number of augmented images for each original image in a given class.
    """
    class_dir = os.path.join(train_dir, class_name)
    images = [img for img in os.listdir(class_dir) if img.endswith(".png")]

    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate and save augmented images
        for _, batch in zip(range(num_images),
                            datagen.flow(x, batch_size=1000, save_to_dir=class_dir, save_prefix='aug_' + class_name,
                                         save_format='png')):
            pass  # This loop will save 'num_images' augmented images for each original image

'''

def gaussian_noise(image):
    """
    Add Gaussian noise to an image.
    """
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)  # Gaussian noise
    noisy_image = np.clip(image + gaussian, 0, 255)  # Add noise and clip the values
    return noisy_image

'''
# ImageDataGenerator with Gaussian noise
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    #horizontal_flip=True,
    #zoom_range=0.1,
    #channel_shift_range=20,  # Randomly shift color channels
    preprocessing_function=gaussian_noise
)
'''

def create_model():
    # Create the NASNetLarge base model (pre-trained on ImageNet)
    base_model = NASNetLarge(
        include_top=False,  # Exclude the top classification layer
        weights='imagenet',  # Load pre-trained weights
        input_shape=input_shape
    )

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for classification
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='leaky_relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='leaky_relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='leaky_relu')(x)  # Additional Dense layer 1
    x = Dropout(0.5)(x)  # Dropout for layer 1
    x = Dense(128, activation='leaky_relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def compile_model(model):
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

def prepare_data(train_dir, validation_dir, test_dir):
    train_datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    zoom_range=0.1,
                    #channel_shift_range=20,  # Randomly shift color channels
                    preprocessing_function=gaussian_noise,
                    rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(331, 331), batch_size=20, class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(331, 331), batch_size=20, class_mode='binary')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(331, 331),
        batch_size=20,
        class_mode='binary',  # or 'categorical' for multi-class
        shuffle=False)

    return train_generator, validation_generator, test_generator


def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # Adjust according to your dataset
        epochs=50,
        validation_data=validation_generator,
        validation_steps=100,  # Adjust according to your dataset
        verbose=2)
    return history


# Create the model

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    return test_loss, test_accuracy


def load_and_preprocess_image(image_path_or_url, from_url=False):
    """
    Load and preprocess an image from a local path or a URL.
    """
    if from_url:
        # Load the image from a URL
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        # Load the image from a local file
        image = Image.open(image_path_or_url)

    # Resize image and convert to RGB (3 channels)
    image = image.resize((331, 331)).convert('RGB')

    # Convert image to array and normalize
    image = img_to_array(image) / 255.0

    # Expand dimensions to match the model's input format
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(model, image_path_or_url, from_url=False):
    """
    Predict whether an image contains fire or no fire.
    """
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path_or_url, from_url)

    # Predict
    prediction = model.predict(processed_image)

    # Interpret prediction
    predicted_class = 'fire' if prediction[0][0] > 0.5 else 'nofire'

    return predicted_class


def get_user_input():
    """
    Get image input from the user, either as a file path or a URL.
    """
    choice = input(
        "Enter '1' to upload a direct image URL (must end with .jpg, .jpeg, .png, etc.) or '2' to upload an image from your computer: ")
    if choice == '1':
        image_url = input("Enter the direct image URL: ")
        if image_url.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return image_url, True
        else:
            print("Please enter a valid image URL that ends with image extension type.")
            return None, None
    elif choice == '2':
        image_path = input("Enter the path to your image file: ")
        return image_path, False
    else:
        print("Invalid input. Please enter '1' or '2'.")
        return None, None


#save_augmented_images('fire')
#save_augmented_images('nofire')

xl_path = make_excel()
data = pd.read_excel(xl_path)

# Now you can manipulate the DataFrame as needed
print(data.head())
train_df = data[data['Split'] == 'train']
print("\n shape of train data: ", train_df.shape)

# Call functions to execute
data_vis()
data_cleaning()

# Display example images from 'fire' and 'nofire' folders
display_images(os.path.join(classification_path, 'train', 'fire'), 'Fire Images')
display_images(os.path.join(classification_path, 'train', 'nofire'), 'No Fire Images')

if not os.path.exists(os.path.join(curr_path, 'model.h5')):
    input_shape = (331, 331, 3)
    model = create_model()

    # Compile the model
    compile_model(model)

    # Prepare data
    train_generator, validation_generator, test_generator = prepare_data(train_dir=train_dir, validation_dir=valid_dir,
                                                                         test_dir=test_dir)
    print("train generator", train_generator)
    print("validation generator", validation_generator)
    # train the model
    history = train_model(model, train_generator, validation_generator)

    # Save the trained model
    model_save_path = os.path.join(curr_path, 'model.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load the saved model
    model = tf.keras.models.load_model(os.path.join(curr_path, 'model.h5'))
    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, test_generator)

# Load the saved model
model = tf.keras.models.load_model(os.path.join(curr_path, 'model.h5'))
# image prediction
image_path_or_url, from_url = get_user_input()
prediction = predict_image(model, image_path_or_url, from_url)
print("Prediction:", prediction)

