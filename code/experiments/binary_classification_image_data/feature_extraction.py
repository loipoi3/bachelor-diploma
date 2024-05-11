import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from code.experiments.binary_classification_image_data.model import Model
from custom_dataset import ChestXRayDataset


def preprocess_labels(labels):
    # Convert labels where 1 or 2 are mapped to 1 and 0 remains as 0
    return np.where(labels > 0, 1, 0)


if __name__ == "__main__":
    # Define the path to dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the dataset with the transform
    train_dataset = ChestXRayDataset("train")
    test_dataset = ChestXRayDataset("test")

    # Use a DataLoader to handle batching
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Load a pre-trained ResNet-50 model and remove the final fully connected layer
    model = Model(3, True)
    checkpoint = torch.load('experiments/binary_classification_image_data/saved_model.pth')
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # Disable gradient computation since we only need forward pass
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get the embeddings
            batch_embeddings = model(images)

            # Reshape embeddings to flatten them out
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)

            labels = preprocess_labels(labels.cpu().numpy())

            # Move the embeddings and labels to CPU and convert them to numpy arrays
            embeddings_list.append(batch_embeddings.cpu().numpy())
            labels_list.extend(labels)

    # Concatenate all embeddings and labels into a single NumPy array
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list).reshape(-1, 1)  # Reshape labels to be a column vector

    # Combine the embeddings and labels into one array
    final_data = np.hstack((all_embeddings, all_labels))

    # Create a DataFrame and save it to a CSV file
    column_names = [f'feature_{i + 1}' for i in range(final_data.shape[1] - 1)] + ['label']
    df = pd.DataFrame(final_data, columns=column_names)
    df.to_csv('datasets/chest_xray/train_embeddings.csv', index=False)

    # Disable gradient computation since we only need forward pass
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get the embeddings
            batch_embeddings = model(images)

            # Reshape embeddings to flatten them out
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)

            labels = preprocess_labels(labels.cpu().numpy())

            # Move the embeddings and labels to CPU and convert them to numpy arrays
            embeddings_list.append(batch_embeddings.cpu().numpy())
            labels_list.extend(labels)

    # Concatenate all embeddings and labels into a single NumPy array
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.array(labels_list).reshape(-1, 1)  # Reshape labels to be a column vector

    # Combine the embeddings and labels into one array
    final_data = np.hstack((all_embeddings, all_labels))

    # Create a DataFrame and save it to a CSV file
    column_names = [f'feature_{i + 1}' for i in range(final_data.shape[1] - 1)] + ['label']
    df = pd.DataFrame(final_data, columns=column_names)
    df.to_csv('datasets/chest_xray/test_embeddings.csv', index=False)
