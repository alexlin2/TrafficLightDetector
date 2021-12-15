from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torch
import numpy as np
from PIL import Image
import yaml

img_dir = "/home/alexlin/traffic_net/dataset_train_rgb/"
labels_dir = "/home/alexlin/traffic_net/dataset_train_rgb/train.yaml"

class TrafficLightDataset(Dataset):
    
    def __init__(self, img_dir, labels_dir, classes : list, long_size=512):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.long_size = long_size
        self.classes = classes
        self.img_paths, self.boxes = self.get_data()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(self.img_dir + img_path[2:]).convert('RGB')
        w, h = img.size
        boxes = self.boxes[index]
        y = {}
        if w > h:
            re_w = self.long_size
            re_h = int(h*re_w/w)
        else:
            re_h = self.long_size
            re_w = int(w*re_h/h)
        re_size = (re_w, re_h)
        x = img.resize(re_size)
        x = F.to_tensor(x)  
        if len(boxes) > 0:
            new_boxes = []
            labels = []
            for box in boxes:
                box['x_min'], box['x_max'] = box['x_min']*re_size[0] / \
                    w, box['x_max']*re_size[0]/w
                box['y_min'], box['y_max'] = box['y_min']*re_size[1] / \
                    h, box['y_max']*re_size[1]/h
                new_boxes.append([box['x_min'], box['y_min'], box['x_max'], box['y_max']])
                labels.append(self.classes.index(box['label']))
            y['boxes'] = torch.tensor(new_boxes, dtype=torch.float)
            y['labels'] = torch.tensor(labels, dtype=torch.long)
        else:
            y['boxes'] = torch.empty((0,4), dtype=torch.float)
            y['labels'] = torch.tensor([0], dtype=torch.long)
            
        return x, y

    def get_data(self):
        img_paths = []
        boxes = []
        with open(self.labels_dir) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            for item in data:
                img_paths.append(item['path'])
                if len(item['boxes']) > 0:
                    n_boxes = []
                    for box in item['boxes']:
                        n_boxes.append(box)
                    boxes.append(n_boxes)
                else:
                    boxes.append([])
        return img_paths, boxes

def collate_fn(batch):
    return tuple(zip(*batch))

        

if __name__ == "__main__":
    dataset = TrafficLightDataset(img_dir, labels_dir, ['background', 'GreenLeft', 'RedStraightLeft', 'RedLeft', 'off', 'GreenStraight', 'GreenStraightRight',
             'GreenStraightLeft', 'RedStraight', 'GreenRight', 'Green', 'Yellow', 'RedRight', 'Red'])
    print(dataset[2][1])
