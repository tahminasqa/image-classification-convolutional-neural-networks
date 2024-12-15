# Image Classification Convolutional Neural Networks
Convolutional Neural Networks (CNNs) are a powerful tool for tasks involving images, such as classifying, detecting objects, and segmenting them. In this project, I build a CNN using TensorFlow/Keras to classify images. The goal of the model is to sort images into specific categories, taking advantage of deep learning techniques to achieve accurate results.

## Features
- Built using TensorFlow and Keras for designing and training the model.
- Includes essential CNN layers like Conv2D, MaxPooling2D, and Dense.
- Incorporates dropout and batch normalization to enhance model performance and reduce the risk of overfitting.

## Requirements
To get this project up and running, make sure you have the following installed on your system, it could be **Google Colab** or **Jupyter Notebook**:

- Python 3.8 or higher
- TensorFlow 2.0 or higher
- NumPy
- Matplotlib
- Keras
- scikit-learn (optional, for evaluating performance)

**Install dependencies with:**
You can easily install all the dependencies by running the following command:

```
bash
pip install tensorflow numpy keras matplotlib scikit-learn
```


## Dataset
The project expects the image dataset to be organized in the following directory structure:

- dataset/
  - 4k_training_data_set/
    - cats/
    - dogs/
  - 1k_test_data_set/
    - cats/
    - dogs/

- 4k_training_data_set/: Contains training images for the model.
  - cats/: Folder containing images of cats.
  - dogs/: Folder containing images of dogs.
- 1k_test_data_set/: Contains test images to evaluate the model.
  - cats/: Folder containing test images of cats.
  - dogs/: Folder containing test images of dogs.

Ensure that the dataset is organized as shown above before running the project or you could use your own format according to the dataset. Here, in my case I have used 4000 data for training and 1000 data for testing.

## Model Architecture
The CNN model consists of the following layers:

- **Convolutional Layers with ReLU Activation:** Extracts features from images using filters and adds non-linearity.
- **MaxPooling Layers:** Reduces the size of the feature maps, making the model faster and more efficient.
- **Flatten Layer:** Converts 2D data into a 1D vector for classification.
- **Dense Layers:** Fully connected layers that classify the features into categories.
- **Dropout Layers:** Randomly disables some neurons to help prevent overfitting.
- **Batch Normalization:** Speeds up training by normalizing data after each layer.

## Usage
1. **Clone the repository**:
   - Run the following command to clone the repo:
     ```bash
     git clone <repository-url>
     ```
   - Navigate to the project folder:
     ```bash
     cd <project-directory>
     ```

2. **Prepare your dataset**:
   - Ensure your dataset is in the required format.

3. **Open the notebook**:
   - Open `CNN_for_Image_Classification_Tahmina.ipynb` in Jupyter Notebook or Google Colab.

4. **Run the cells**:
   - Execute the cells sequentially to:
     - Load and preprocess the dataset.
     - Build and compile the CNN model.
     - Train the model and evaluate its performance.


## Results
The key findings from the image classification project are detailed in a post on Medium. You can read the full post by following the link below:

[Read the full post on Medium](https://medium.com/@tahminasqa/developing-an-image-classification-model-with-convolutional-neural-networks-cnns-5c4cfb6d89ae)

## Acknowledgments
I would like to express my gratitude to the following:
- **TensorFlow/Keras** for providing a powerful and flexible deep learning framework.
- **The dataset providers**

## License
Specify your license here (e.g., MIT License).
