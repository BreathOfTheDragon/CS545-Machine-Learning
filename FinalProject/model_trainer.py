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

torch.cuda.empty_cache()
num_epochs = 10


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


def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        
        torch.cuda.empty_cache()

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('---------------------')

        c = datetime.now()
        current_time = c.strftime('%H:%M:%S')
        print('Current Time is:', current_time)
        print('---------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

   
    
data_dir = f'./dog-cat-full-dataset-master/data/train'

full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

train_dataset_percentage = 0.90

train_size = int(train_dataset_percentage * len(full_dataset))

val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=256)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=256)



dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models = [
#           models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1),
#           models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
#           models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
#           ]

models = [
            models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
          ]



# model_names = ["GoogleNet", "ResNet50", "VGG19"]

model_names = ["ResNet50"]


# train phase:

for i, model_ft in enumerate(models):

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model_ft = nn.DataParallel(model_ft)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0001, weight_decay=1e-5)


    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=num_epochs)

    models_dir = './models'
    os.makedirs(models_dir, exist_ok=True)

    model_save_path = os.path.join(models_dir, f"TransferModel_{model_names[i]}_{num_epochs}.pth")
    torch.save(model_ft.state_dict(), model_save_path)

    print(f"Model saved at: {model_save_path}")



    # validation phase


    model_ft.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    reports_dir = './Reports'
    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, f"classification_report_val_{model_names[i]}_{num_epochs}.txt")
    with open(report_path, 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))


    confusion_path = os.path.join(reports_dir, f"confusion_matrix_val_{model_names[i]}_{num_epochs}.png")
    plot_confusion_matrix(cm, class_names, confusion_path)

    print(f"Classification report saved at: {report_path}")
    print(f"Confusion matrix saved at: {confusion_path}")

