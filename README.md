# DL_final_project
# Wildfire Prediction with Deep Learning

This project focuses on developing a deep learning model for predicting wildfires using satellite images and weather data.

## Problem Statement

Wildfires pose a major threat to ecosystems, communities, and infrastructure around the world. Their increasing occurrence and intensity necessitate the development of reliable prediction systems for early detection and intervention.

## Project Goals

* Develop a deep learning model for accurate wildfire prediction.
* Utilize a combination of satellite images and weather data for improved prediction accuracy.
* Implement the model in a real-time application for early warning and response efforts.

## Methodology

1. **Data Acquisition:** Collect and pre-process satellite images and weather data from reliable sources.
2. **Model Design:** Develop and train a deep learning model using appropriate architectures and hyperparameter tuning.
3. **Model Evaluation:** Evaluate the performance of the model on unseen data and compare it to existing methods.
4. **Web Application Development:** Create a user-friendly web interface for easy access and utilization of the model.

## Data Download
* We have directly downloaded our dataset in to the ubuntu server using kaggle api commands, below shows the code to download the dataset and folder manipulations according to the code.
* from google.colab import files
files.upload()  # Use this to upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
* ! kaggle datasets download -d elmadafri/the-wildfire-dataset
* ! kaggle datasets download -d anamibnjafar0/flamevision
* ! unzip the-wildfire-dataset.zip
* ! unzip flamevision.zip
* ! mv /content/Classification1/the_wildfire_dataset/test/fire/Both_smoke_and_fire/* /content/Classification1/the_wildfire_dataset/test/fire
* ! mv /content/Classification1/the_wildfire_dataset/test/fire/Smoke_from_fires/* /content/Classification1/the_wildfire_dataset/test/fire
* ! mv /content/Classification1/the_wildfire_dataset/test/nofire/Fire_confounding_elements/* /content/Classification1/the_wildfire_dataset/test/nofire
* ! mv /content/Classification1/the_wildfire_dataset/test/nofire/Forested_areas_without_confounding_elements/* /content/Classification1/the_wildfire_dataset/test/nofire
* ! mv /content/Classification1/the_wildfire_dataset/test/nofire/Smoke_confounding_elements/* /content/Classification1/the_wildfire_dataset/test/nofire

* ! mv /content/flames/flamesvision/* /content
* ! rm -r /content/flames
* Similary for all other train and val folder manipulation can be done for both the dataset and merged in to single folder named classification.

## Execution hierarcy
1. **model_pred_resnet50.py**
This script is used for training a wildfire image classification model using the ResNet50 architecture. It includes steps for setting up the environment, downloading and preprocessing the dataset, data augmentation, model definition, training, and evaluation with confusion matrix visualization. It also contains functions for user input and prediction on new images.

2. **eda.py**
This script is focused on exploratory data analysis (EDA) of the image dataset. It includes creating an Excel file with image metadata, visualizations of data distribution, and basic data cleaning operations like handling missing values, duplicates, and ensuring data consistency.

3. **model_pred_vgg16.py**
This script implements a VGG16-based image classification model. It outlines model training, saving, and evaluation with detailed accuracy and loss plots. The VGG16 model is determined to be the best-performing model for the given task.

## Model Selection

* **Basic CNN:** Demonstrated reasonable performance but lacked sufficient complexity for accurate prediction.
* **ResNet50:** Achieved high accuracy but suffered from overfitting, limiting itsgeneralizability.
* **VGG16:** Emerged as the best performing model, achieving high accuracy (90%) and low loss, indicating robust prediction capabilities.

## Web Application Features

* Upload images for real-time wildfire prediction.
* Visualize predicted fire locations and severity levels.
* Access historical wildfire data for analysis and trend identification.
* Provide user-friendly interface for seamless interaction with the model.

## Project Outcomes

* Developed a highly accurate deep learning model for wildfire prediction.
* Implemented the model in a user-friendly web application for real-time use.
* Demonstrated the potential of deep learning technology for proactive wildfire management.

## Future Work

* Integrate additional data sources, such as social media and sensor networks, for further enhancement.
* Explore ensemble learning techniques to combine the strengths of different models.
* Develop explainability methods to understand model predictions and build trust.

The datasets used are:

- [FlameVision Dataset](https://www.kaggle.com/datasets/anamibnjafar0/flamevision/code)
- [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/data)

## Authors

- Mahikshit Kurapati
- Pooja Chandrashekara
- Mohammad Kanu


## Version History

- 0.1
    - Initial Release

## Acknowledgments

- Special thanks to Amir Jafari, Professor, Department of Data Science, for guidance and mentorship.
- Recognition to Kaggle and the dataset contributors.

## References

* https://www.ijnrd.org/papers/IJNRD2305193.pdf
* https://github.com/artefactory/streamlit_prophet
* https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0



## Version History

- 0.1
    - Initial Release
