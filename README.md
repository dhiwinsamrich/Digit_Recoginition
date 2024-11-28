# Digit Recognition Project using MNIST Dataset

## Project Overview
This project implements a neural network-based digit recognition system using the MNIST dataset. It includes model training, evaluation, and an interactive GUI for drawing and predicting handwritten digits.

## Features
- Neural network digit classification
- Machine learning model training on MNIST dataset
- Interactive drawing and prediction interface
- Model optimization with regularization and early stopping

## Technology Stack
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Tkinter

## Model Architecture
- Input Layer: 784 neurons (28x28 pixel flattened image)
- First Hidden Layer: 32 neurons with ReLU activation
- Second Hidden Layer: 32 neurons with ReLU activation
- Dropout Layer: 50% dropout rate for preventing overfitting
- Output Layer: 10 neurons with softmax activation (0-9 digit classification)

## Key Components

### Data Preprocessing
- Normalize pixel values to range [0, 1]
- Reshape images to flat 784-dimensional vectors
- Convert labels to categorical format

### Model Training
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Early Stopping: Monitors validation loss
- L2 Regularization to prevent overfitting

### GUI Features
- Interactive canvas for drawing digits
- Real-time digit prediction
- Clear canvas functionality

## Performance Metrics
- Model trained on MNIST dataset
- Evaluation metric: Accuracy
- Model saved after training for future use

## How to Run
1. Install required dependencies
2. Run the script
3. Draw a digit in the GUI canvas
4. Click "Predict" to see the predicted digit

## Dependencies
- tensorflow
- numpy
- matplotlib
- pillow
- tkinter

## Model Training Process
1. Load MNIST dataset
2. Preprocess data
3. Define neural network architecture
4. Compile model
5. Train with early stopping
6. Save trained model
7. Evaluate model performance

## Potential Improvements
- Increase model complexity
- Add data augmentation
- Experiment with different architectures
- Implement model uncertainty estimation

## License
MIT License

## Author
DHIWIN SAMRICH
