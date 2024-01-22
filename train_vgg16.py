import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from tqdm import tqdm
from datetime import datetime

from models import VGG16
from datasets import OdometerTypeDataset

# tensorboard monitoring
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(train_loader, model, criterion, optimizer, device):
    running_loss = 0.0
    last_loss = 0.0

    for i in range(len(train_loader)):
        inputs, labels = train_loader[i]
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.0

    return last_loss

def train(train_loader, model, criterion, optimizer, device, epochs, val_loader=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        model.train()
        avg_loss = train_one_epoch(train_loader, model, optimizer, device)

        running_vloss = 0.0

        model.eval()
        with torch.no_grad():
            for j in range(len(val_loader)):
                vinputs, vlabels = val_loader[j]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss
        
        avg_vloss = running_vloss / (j + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}.pth'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

def evaluate(test_loader, model, device, metrics=[]):
    accuracy, precision, recall, f1 = metrics[0], metrics[1], metrics[2], metrics[3]

    preds = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_loader)):
            inputs, labels = test_loader[i]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds.extend(outputs)
            y_true.extend(labels)

    preds = torch.Tensor(preds)
    y_true = torch.Tensor(y_true)

    print(f'Accuracy: {accuracy(preds, y_true)}')
    print(f'Precision: {precision(preds, y_true)}')
    print(f'Recall: {recall(preds, y_true)}')
    print(f'F1-Score: {f1(preds, y_true)}')

def main():
    # set hyperparameters
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    INPUT_SIZE = (227, 227)

    # set device cuda if available, else set cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize metrics
    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()

    # create the model
    model = VGG16().to(device)

    # leave top layers as trainable, freeze others for transfer learning
    for p in model.model.parameters():
        p.requires_grad = False

    for c in list(model.model.children())[0][-5:]:
        for p in c.parameters():
            p.requires_grad = True

    for c in list(model.model.children())[2]:
        for p in c.parameters():
            p.requires_grad = True

    # binary crossentropy loss function
    # ADAM optimizer
    # resize transformation
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    transform = transforms.Resize(INPUT_SIZE)

    # Load train, validation and test datasets
    train_loader = OdometerTypeDataset('dataset/train.txt', 'trodo-v01/images', image_size=INPUT_SIZE, transform=transform)
    val_loader = OdometerTypeDataset('dataset/val.txt', 'trodo-v01/images', image_size=INPUT_SIZE, transform=transform)
    test_loader = OdometerTypeDataset('dataset/test.txt', 'trodo-v01/images', image_size=INPUT_SIZE, transform=transform)

    train(train_loader, model, criterion, optimizer, device, EPOCHS, val_loader=val_loader)
    evaluate(test_loader, model, device, metrics=[accuracy, precision, recall, f1])

if __name__ == '__main__':
    main()