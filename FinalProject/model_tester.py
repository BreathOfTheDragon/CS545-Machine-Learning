import os
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import time
from PIL import Image
from datetime import datetime
import torchvision.transforms.functional as F
from torchinfo import summary


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    plt.savefig(save_path)
    plt.close()  
    

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        image = F.pad(image, padding)
        
        return F.resize(image, (224, 224))


data_transforms = transforms.Compose([
    transforms.Resize(224),  
    SquarePad(),            
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])



test_data_dir = f'./dog-cat-full-dataset-master/data/test'

num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = datasets.ImageFolder(test_data_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=16)


# load phase: 



model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(test_dataset.classes))

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

model.load_state_dict(torch.load(f'./models/TransferModel_ResNet50_{num_epochs}.pth'))



# test phase: 

model.eval()

all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

cm_test = confusion_matrix(all_test_labels, all_test_preds)
print('Test Confusion Matrix:')
print(cm_test)

print('\nTest Classification Report:')
print(classification_report(all_test_labels, all_test_preds, target_names=test_dataset.classes))

reports_dir = './Reports'
os.makedirs(reports_dir, exist_ok=True)


test_confusion_path = os.path.join(reports_dir, f"confusion_matrix_test_ResNet50_{num_epochs}.png")
plot_confusion_matrix(cm_test, test_dataset.classes, test_confusion_path)

test_report_path = os.path.join(reports_dir, f"classification_report_test_ResNet50_{num_epochs}.txt")
with open(test_report_path, 'w') as f:
    f.write(classification_report(all_test_labels, all_test_preds, target_names=test_dataset.classes))

print(f"Test Classification report saved at: {test_report_path}")
print(f"Test Confusion matrix saved at: {test_confusion_path}")


