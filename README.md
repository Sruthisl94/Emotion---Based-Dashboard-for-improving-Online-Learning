## Tilte: 
Emotion based dashboard for improving online learning
## Objective: 
Creating interactive dashboard that assesses student's emotion during online learning and providing educators with precise, realtime emotional insights to tailor their teaching methodologies effectively.

## Methodology

1. Dataset

Fer2013 and CK+ dataset from Kaggle is combined and images having 7 emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral are used in this project.

2. Preprocessing and Augmentation

The images are rescaled to normalize pixel values between 0 and 1.

The training data is augmented to increase diversity and prevent overfitting using:

Rotation

Width and height shifts

Zoom

Horizontal flip

All images are resized to (48, 48) pixels and converted to grayscale.

3. Model Architecture
A CNN model is constructed with the following layers:

Two convolutional layers with ReLU activation

MaxPooling layers after each convolution

Dropout layers (to prevent overfitting)

A fully connected dense layer

Final output layer with Softmax activation (for 7 classes)

Loss Function: Categorical Crossentropy

Optimizer: Adam with learning rate 1e-4

4. Training
The model is trained for up to 90 epochs with:

EarlyStopping (monitors validation loss)

ReduceLROnPlateau (adjusts learning rate on plateau)

ModelCheckpoint (saves best-performing model)

5. Evaluation
The model is evaluated using:

Accuracy

Confusion Matrix

Classification Report (precision, recall, F1-score)

Final model achieved approximately 80% validation accuracy after tuning.

# Results and Disussions
For real time monitoring of student emotions during an online platfoorm, an interactive dashboard has been built using Streamlit. The architecture of the interface is shown below:
In the dashboard the students can log in using their ID, and their face is captured via the webcam. The image is then processed, and with the trained model, the dashboard can predict the emotion. If a students appears sad, the teacher receives an alert to enhance the student's engagement by adjusting the teaching method. The dashboard also stores the information on all the students accesses. The graphical outputs, including bar graphs showing the number of studets under each emotion category, assst  in identifying the overall class performance.

# Future work
The dashboard can be connected to Zoom or Google Meet to automatically capture student facial expressions during live online classes, enabling real-time emotion detection and feedback without requiring students to access the dashboard separately.

## Usage

1. Clone this repository.
2. Install the required dependencies.
3. Or directly access the hosted dashboard here:  
   [🎯 Streamlit App](https://emotion-based-dashboard-analysis.streamlit.app/)
