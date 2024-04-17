import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from custom_dataset import CatsAndDogsDataset

if __name__ == "__main__":
    # Define the path to dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path('datasets/cats_vs_dogs')

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of ResNet
        transforms.ToTensor(),          # Convert image to a float tensor and scale to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Instantiate the dataset with the transform
    dataset = CatsAndDogsDataset(root_dir=dataset_path, transform=transform)

    # Use a DataLoader to handle batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load a pre-trained ResNet-50 model and remove the final fully connected layer
    model = models.resnet50(pretrained=True).to(device)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()  # Set to evaluation mode

    # Disable gradient computation since we only need forward pass
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get the embeddings
            batch_embeddings = feature_extractor(images)

            # Reshape embeddings to flatten them out
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)

            # Move the embeddings and labels to CPU and convert them to numpy arrays
            embeddings_list.append(batch_embeddings.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Concatenate all embeddings and labels into a single NumPy array
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list).reshape(-1, 1)  # Reshape labels to be a column vector

    # Combine the embeddings and labels into one array
    final_data = np.hstack((all_embeddings, all_labels))

    # Create a DataFrame and save it to a CSV file
    column_names = [f'feature_{i + 1}' for i in range(final_data.shape[1] - 1)] + ['label']
    df = pd.DataFrame(final_data, columns=column_names)
    df.to_csv('datasets/cats_vs_dogs/embeddings.csv', index=False)
