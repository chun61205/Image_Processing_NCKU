import torch
import os

import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import Lambda
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.imagePaths = []
        self.classes = ['Cat', 'Dog']

        for className in self.classes:
            classDir = os.path.join(self.rootDir, className)
            self.imagePaths += [os.path.join(classDir, file)
                                 for file in os.listdir(classDir)
                                 if file.endswith('.jpg')]

        self.labels = [self.classes.index(os.path.basename(os.path.dirname(path)))
                       for path in self.imagePaths]

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = Image.open(imagePath)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        label = self.labels[idx]


        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':

    epochs = 20
    modelPath = 'resnet50.pth'

    transformWithoutRandomErasing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    transformWithRandomErasing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing()
    ])

    model = resnet50(pretrained=False)
    numFeatures = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(numFeatures, 1),
        nn.Sigmoid()
    )

    trainDataset = CatDogDataset(rootDir='/content/drive/MyDrive/dataset/training_dataset', transform=transformWithoutRandomErasing)
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    valDataset = CatDogDataset(rootDir='/content/drive/MyDrive/dataset/validation_dataset', transform=transformWithoutRandomErasing)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available!")
        device = torch.device("cpu")
    weight_dtype = torch.float32
    model.to(device, dtype=weight_dtype)
    summary(model, (3, 224, 224))

    optimizerParams = {
        'params': model.parameters(),
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0
    }

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(**optimizerParams)

    trainLosses = []
    valLosses = []
    trainAccuracies = []
    valAccuracies = []

    bestAccuracy = 0.0
    bestEpoch = 0

    patience = 30
    bestLoss = None
    earlyStopCounter = 0

    for epoch in range(epochs):
        model.train()
        trainRunningLoss = 0.0
        trainTotal = 0
        trainCorrect = 0
        for inputs, labels in trainLoader:
            optimizer.zero_grad()

            outputs = model(inputs.to(device, dtype=weight_dtype))
            outputs = outputs.squeeze()
            loss = criterion(outputs.squeeze(), labels.to(device).float())

            loss.backward()
            optimizer.step()

            prediction = (outputs >= 0.5).float()
            trainTotal += labels.size(0)
            trainCorrect += (prediction == labels.to(device)).sum().item()

            trainRunningLoss += loss.item()

        trainLoss = trainRunningLoss / len(trainLoader)
        trainLosses.append(trainLoss)
        trainAccuracy = trainCorrect / trainTotal
        trainAccuracies.append(trainAccuracy)


        model.eval()
        valRunningLoss = 0.0
        valTotal = 0
        valCorrect = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                outputs = model(inputs.to(device))
                outputs = outputs.squeeze()
                loss = criterion(outputs.squeeze(), labels.to(device).float())
                valRunningLoss += loss.item()

                prediction = (outputs >= 0.5).float()
                valTotal += labels.size(0)
                valCorrect += (prediction == labels.to(device)).sum().item()

        valLoss = valRunningLoss / len(valLoader)
        valLosses.append(valLoss)
        valAccuracy = valCorrect / valTotal
        valAccuracies.append(valAccuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {trainLoss}')
        print(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {trainAccuracy}')
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {valLoss}')
        print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {valAccuracy}')

        if bestLoss is None:
            bestLoss = valLoss
        elif valLoss < bestLoss:
            bestLoss = valLoss
            earlyStopCounter = 0
            torch.save(model.state_dict(), modelPath)
        else:
            earlyStopCounter += 1
            if earlyStopCounter >= patience:
                break

    print('Finished Training')
    torch.save(model.state_dict(), modelPath)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), trainLosses, label='Train Loss')
    plt.plot(range(1, epochs+1), valLosses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), trainAccuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), valAccuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('Loss and Accuracy.jpg', bbox_inches='tight')