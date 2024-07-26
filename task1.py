import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from ImprovedNN import ImprovedNN

def predict_single_image(model_path, image_path, device):
    # Load the model
    model = ImprovedNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64
        transforms.Normalize((0.5,), (0.5,))  # Normalize to 0-1
    ])
    
    image = read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    
    return pred.item()

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "improved_mnist_model.pt"  # Path to your saved model
image_path = "image/9_2991_E.jpg"  # Path to the image you want to predict

predicted_digit = predict_single_image(model_path, image_path, device)
print(f"The predicted digit is: {predicted_digit}")
