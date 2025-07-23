## Tilte: 
Emotion based dashboard for improving online learning
## Objective: 
Creating interactive dashboard that assesses student's emotion during online learning and providing educators with precise, realtime emotional insights to tailor their teaching methodologies effectively.

## Methodology

1.Dataset:
Fer2013 and CK+ dataset from Kaggle is combined and images havign 7 emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral are used in this project.
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
For real time monitoring of student emotions during online platfoorm, an interactive dashboard has been created using streamlit. The architecture of the interface is shown below:

