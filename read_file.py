from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class Read_file(Dataset):
    def __init__(self, rootdir, transform = None):
        super().__init__()
        self.image_path, self.label = [], []
        self.transform = transform
        class_to = {"cats" : 0, "dogs" : 1}


        for labe, item in class_to.items():
            class_dir = os.path.join(rootdir, labe)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(".jpg"):
                        self.image_path.append(os.path.join(class_dir, filename))
                        self.label.append(item)
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label[index]

