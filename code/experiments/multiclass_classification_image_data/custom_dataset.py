from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class MulticlassImageDataset(Dataset):
    """
    A dataset class for loading and transforming images for model training or testing.

    Attributes:
        class_to_idx (dict): A mapping from class names to integer labels.
        ...
    """

    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset object for multiclass classification, listing all image
        files and storing the transformation function provided.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = [p for p in self.root_dir.glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        # Automatically build a mapping from class names to integers
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(sorted(self.root_dir.iterdir()))}

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Here we read the class name and translate it to a numeric label
        label = self.class_to_idx[image_path.parent.name]
        return image, label
