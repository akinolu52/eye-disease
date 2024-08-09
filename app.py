import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

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

    print('ok -> ',predicted.item())
    return classes[predicted.item()]


st.title("Eye Image Classification")

uploaded_file = st.file_uploader("Upload an image of an eye", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Classify Image'):
        result = classify_eye(vit_model, image)
        st.write(f"Prediction: {result}")
