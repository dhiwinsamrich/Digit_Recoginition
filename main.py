import tkinter as tk  # Import tkinter with alias tk
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import tensorflow as tf  # Import TensorFlow

# Load the saved model
model = tf.keras.models.load_model("D:\\Internship\\Zidio Internship\\Digit Recoginition\\model_with_optimizer.h5")

# Define preprocessing and prediction functions
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = image.convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (white background, black digit)
    image = np.array(image).astype('float32') / 255  # Normalize pixel values
    image = image.reshape(1, 28 * 28)  # Flatten the image to match the model input
    return image

def predict_digit(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return np.argmax(prediction)

# Tkinter GUI Application
class DrawAndPredictApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit")

        # Canvas for drawing
        self.canvas = tk.Canvas(master, width=200, height=200, bg="white")
        self.canvas.pack()

        # Buttons for prediction and clearing
        self.button_predict = tk.Button(master, text="Predict", command=self.make_prediction)
        self.button_predict.pack()

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        # Bind mouse movement to paint function
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create a blank image to draw on
        self.image = Image.new("L", (200, 200), 255)  # 200x200 white canvas
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        # Draw a small circle at the mouse position
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black", outline="black")
        self.draw.ellipse((x-5, y-5, x+5, y+5), fill="black", outline="black")

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def make_prediction(self):
        # Predict the digit and print it
        predicted_digit = predict_digit(self.image)
        print(f'Predicted Digit: {predicted_digit}')


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawAndPredictApp(root)
    root.mainloop()
