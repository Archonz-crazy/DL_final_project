import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt
#import cv2 as cv
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from keras.utils import to_categorical
import os

cwd = os.getcwd()
curr_path = os.path.dirname(cwd)
classification_path = os.path.join(curr_path, 'Wildfire dataset'+'/'+'the_wildfire_dataset'+'/'+'the_wildfire_dataset')
code_path = os.path.join(curr_path, 'code')
print("model path", os.path.join(code_path, 'model.h5'))
train_dir = os.path.join(classification_path, 'train')
valid_dir = os.path.join(classification_path, 'val')
test_dir = os.path.join(classification_path, 'test')

test_classes = os.listdir(test_dir)
print(test_classes)

input_shape = (224,224,3)
num_classes = 2

trainGenertor = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 10,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    shear_range = 0.2,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    dtype = 'float32'
)
valGenertor = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    dtype = 'float32'
)

testGenertor = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    dtype = 'float32'
)

train_data = trainGenertor.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 16,
    class_mode = 'categorical'
)

val_data = valGenertor.flow_from_directory(
    valid_dir,
    target_size = (224,224),
    batch_size = 16,
    class_mode = 'categorical'
)

test_data = testGenertor.flow_from_directory(
    test_dir,
    target_size = (224,224),
    batch_size = 16,
    class_mode = 'categorical',
    shuffle = False
)

if os.path.exists(os.path.join(code_path, 'model.h5')):
    VGG16_model = VGG16(
        include_top = False,
        weights="imagenet",
        input_shape = input_shape
    )
    for layer in VGG16_model.layers :
        layer.trainable = False

    from keras import activations
    model = Sequential()
    model.add(VGG16_model)
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(num_classes,activation = 'softmax'))
    model.summary()

    model.compile(
        optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('model.h5',save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]


    results = model.fit(train_data,validation_data=val_data,epochs=5,verbose = 1)
    loss, acc = model.evaluate(test_data,verbose = 1)
    model_save_path = os.path.join(code_path, 'model.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Load the saved model
# Evaluate the model
model = tf.keras.models.load_model(os.path.join(code_path, 'model.h5'))
loss, acc = model.evaluate(test_data, verbose=1)

import seaborn as sns
predictions_prob = model.predict(test_data)
predictions = np.argmax(predictions_prob, axis=1)
true_labels = test_data.classes
report = classification_report(true_labels, predictions)
print(report)

conf_mat = confusion_matrix(true_labels,predictions)
sns.heatmap(conf_mat ,fmt='g',annot = True , cmap='Blues' , xticklabels=test_classes , yticklabels = test_classes,)
plt.xlabel('Predictions')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.xticks(rotation = 45)
plt.show()

res = results.history
train_acc = res['accuracy']
val_accuracy = res['val_accuracy']
epochs = range(1, len(train_acc) + 1)

line1 = plt.plot(epochs, val_accuracy, label = 'Validation/Test Accuracy')
line2 = plt.plot(epochs, train_acc, label = 'Training Accuracy')

plt.setp(line1, linewidth = 1.8, marker = 'o', markersize = 6.5)
plt.setp(line2, linewidth = 1.8, marker = 's', markersize = 5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

## plot of val accuracy, train accuracy
res = results.history
train_acc = res['accuracy']
val_accuracy = res['val_accuracy']
epochs = range(1, len(train_acc) + 1)

line1 = plt.plot(epochs, val_accuracy, label = 'Validation/Test Accuracy')
line2 = plt.plot(epochs, train_acc, label = 'Training Accuracy')

plt.setp(line1, linewidth = 1.8, marker = 'o', markersize = 6.5)
plt.setp(line2, linewidth = 1.8, marker = 's', markersize = 5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

## plot of val loss and train loss

res = results.history
train_loss = res['loss']
val_loss = res['val_loss']
epochs = range(1, len(train_loss) + 1)

line1 = plt.plot(epochs, val_loss, label = 'Validation/Test Loss')
line2 = plt.plot(epochs, train_loss, label = 'Training Loss')

plt.setp(line1, linewidth = 1.8, marker = 'o', markersize = 6.5)
plt.setp(line2, linewidth = 1.8, marker = 's', markersize = 5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()