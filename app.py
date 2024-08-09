import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

# import tensorflow as tf


print(timm.__version__)

# Define the classes
classes = [
    "Diabetes",
    "Normal Fundus",
    "Cataract"
]

models_path = {
    'Vit-Model': 'models/vit_base_patch16_224_best.pth',
    'Efficientnet': 'models/efficientnet_b0_best.pth',
    'Densenet': 'models/densenet121_best.pth',
    'Resnet50': 'models/resnet50_best.pth'
}


# def classify_eye(model, model_name, image_path, classes):
#     # Define the mean and std for normalization
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
# 
#     # Define image transformations
#     image_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
#     ])
# 
#     model = model.eval()
#     image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
#     image = image_transforms(image).float()
#     image = image.unsqueeze(0)  # Add batch dimension
# 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     image = image.to(device)
# 
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output.data, 1)
# 
#     print(f"{model_name} -> ", classes[predicted.item()])


@st.cache_resource
def load_vit_model():
    # Initialize the Vision Transformer (ViT) model architecture
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=False)
    num_classes = len(classes)
    vit_model.head = torch.nn.Linear(vit_model.head.in_features, num_classes)

    # Load the saved state dictionary
    vit_model.load_state_dict(torch.load(models_path.get('Vit-Model'), map_location=torch.device('cpu')))

    vit_model.eval()
    print('Vit-Model loaded successfully')
    return vit_model


# Load the model
vit_model = load_vit_model()


def classify_eye(model, image):
    # Define the mean and std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Preprocess the image
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    image = image_transforms(image).unsqueeze(0)  # Add batch dimension

    # Move model and image to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return classes[predicted.item()]


# 
# def classify_image(_image):
#     # # Preprocess the image as required by your model
#     # img = np.array(_image.resize((224, 224)))  # Resize image to the size expected by the model
#     # img = img / 255.0  # Normalize if required
#     # img = np.expand_dims(img, axis=0)  # Add batch dimension
#     # 
#     # # Predict the class
#     # _predictions = model.predict(img)
#     # return _predictions
#     classify_eye(vit_model, "Vit-Model", _image, classes)


st.title("Eye Image Classification")

uploaded_file = st.file_uploader("Upload an image of an eye", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image', use_column_width=True)
    # st.write("")
    # 
    # if st.button('Classify Image'):
    #     result = classify_eye(vit_model, image)
    #     st.write(f"Prediction: {result}")
    image = Image.open(uploaded_file)

    # Display a smaller version of the image
    thumbnail = image.copy()
    thumbnail.thumbnail((300, 300))  # Resize to thumbnail size
    st.image(thumbnail, caption='Uploaded Image (click to expand)', use_column_width=False)

    # Button to view the full-size image
    if st.button('View Full Image'):
        st.image(image, caption='Full-Size Image', use_column_width=True)

    if st.button('Classify'):
        result = classify_eye(vit_model, image)
        st.write(f"Prediction: {result}")
