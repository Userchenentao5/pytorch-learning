from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataSet(Dataset):

    def __init__(self, root_dir: str, label_dir: str):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_paths = os.listdir(self.path)

    def __getitem__(self, idx: int):
        image_name = self.image_paths[idx]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        image = Image.open(image_item_path).convert('RGB')
        return image, self.label_dir

    def __len__(self):
        return len(self.image_paths)

