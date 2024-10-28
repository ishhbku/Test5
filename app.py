# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from torchvision import models
import os

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the image preprocessing function
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define the prediction function
def predict(image):
    try:
        image = Image.fromarray(image)  # Convert the image from NumPy array to PIL format
        image = transform_image(image).to(device)  # Preprocess the image and send to device

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(image)  # Perform inference
            _, predicted = torch.max(outputs, 1)  # Get the predicted class index

        # List of class names
        class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
                       'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
                       'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
                       'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'other']
        return class_names[predicted.item()]
    except Exception as e:
        return str(e)  # Return the error message if something goes wrong

# Initialize the DenseNet-201 model
weights = models.DenseNet201_Weights.IMAGENET1K_V1
model = models.densenet201(weights=weights)

# Freezing the parameters of the model to prevent backpropagation through them
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier layer for 21 classes
model.classifier = nn.Sequential(
    nn.Linear(1920, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 21),  # Adjusted for 21 classes
)

# Load the saved model weights
checkpoint_path = "solution.pth"
state_dict = torch.load(checkpoint_path, map_location=device)
state_dict.pop('classifier.2.weight', None)
state_dict.pop('classifier.2.bias', None)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()  # Set model to evaluation mode

# Set up Gradio interface
interface = gr.Interface(fn=predict,
                         inputs=gr.Image(type="numpy"),
                         outputs="label",
                         title="Food-101 Image Classifier",
                         description="Upload an image and get a classification result for 21 food classes.")

# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
