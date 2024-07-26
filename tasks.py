import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from MultiTaskNN import MultiTaskNN
from hw1 import load_and_preprocess_data



def predict_single_image(model_path, image_path, device, author_to_idx):
    """
    Predicts the digit and author for a single image using a pre-trained multi-task model.
    
    Args:
        model_path (str): Path to the saved model.
        image_path (str): Path to the image file.
        device (torch.device): Device to perform computation on (CPU or CUDA).
        author_to_idx (dict): Mapping from author names to indices.
    
    Returns:
        tuple: Predicted digit and author name.
    """
    # Load the model
    model = MultiTaskNN(len(author_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define the transformation to be applied on the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Read and preprocess the image
    image = read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        image = image.to(device)
        digit_output, author_output = model(image)
        digit_pred = digit_output.argmax(dim=1, keepdim=True)
        author_pred = author_output.argmax(dim=1, keepdim=True)
    
    # Convert author index back to author name
    idx_to_author = {idx: author for author, idx in author_to_idx.items()}
    return digit_pred.item(), idx_to_author[author_pred.item()]

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "multi_task_model.pt"  # Path to your saved model
image_path = "image/5_2335_D.jpg"  # Path to the image you want to predict

# Load author to index mapping
img_dir = 'image'  # Directory containing your images
_, _, author_to_idx = load_and_preprocess_data(img_dir)

# Predict digit and author
predicted_digit, predicted_author = predict_single_image(model_path, image_path, device, author_to_idx)
print(f"The predicted digit is: {predicted_digit}")
print(f"The predicted author is: {predicted_author}")
