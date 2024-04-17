from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


# Define dataset class
class CatsAndDogsDataset(Dataset):
    """
    A dataset class for loading and transforming images of cats and dogs for model training or testing.

    Attributes:
        root_dir (str or Path): The root directory where the images are stored.
        transform (callable, optional): An optional transform to be applied on a sample.
        images (list of Path): List of image file paths under the root directory.

    Args:
        root_dir (str or Path): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset object, listing all image files in the specified directory
        and storing them along with any transformation function provided.

        Parameters:
            root_dir (str or Path): The root directory from which to load the images.
            transform (callable, optional): A function/transform that takes in an image
                and returns a transformed version. E.g, transformations from the torchvision library.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = [p for p in self.root_dir.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves an image by index idx from the dataset and applies the specified transformations.

        Args:
            idx (int): The index of the image file to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is the
                   binary class label (1 for dog, 0 for cat).
        """
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')  # Convert the image to RGB

        if self.transform:
            image = self.transform(image)  # Apply the transform

        label = 1 if image_path.parent.name == 'Dog' else 0
        return image, label
