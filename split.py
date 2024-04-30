import os
import json
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, test_size=0.2):
    categories = ['Bengin cases', 'Normal cases', 'Malignant cases']
    data = {}
    
    for category in categories:
        path = os.path.join(base_dir, category)
        images = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(".jpg")]
        data[category] = images
    
    train_data = {}
    test_data = {}
    
    for category, images in data.items():
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        train_data[category] = train_imgs
        test_data[category] = test_imgs
    
    split_info = {'train': train_data, 'test': test_data}
    with open(os.path.join(base_dir, 'split_info.json'), 'w') as fp:
        json.dump(split_info, fp, indent=4)

if __name__ == '__main__':
    split_dataset('./dataset')