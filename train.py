from dataset import AnimalDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import argparse
from torchvision.models import resnet18, ResNet18_Weights

"""
    Train the model using resnet18
"""

#This is the argument to change the hyperparameters of the model
def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--data-path", "-d", type = str, default = "datasets/animals", help = "path to dataset")
    parser.add_argument("--log-path", "-o", type = str, default = "animal_CNN/tensorboard", help = "tensorboard folder")
    parser.add_argument("--checkpoint-path", "-c", type = str, default = "animal_CNN/checkpoints", help = "checkpoint folder")
    parser.add_argument("--image-size", "-i", type = int, default = 224, help = "common size of all images")
    parser.add_argument("--batch-size", "-b", type = int, default = 32, help = "batch size of training procedure")
    parser.add_argument("--num-epochs","-e", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--lr", "-l", type = float, default = 1e-3, help = "learning rate of optimizer")
    parser.add_argument("--resume-training", "-r", type = bool, default = False, help = "Continue to train from the last model or not")

    args = parser.parse_args()
    return args

#This function is used to plot the confusion matrix in the tensorboard
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="hsv")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

#This function are steps to train the model
def train(args):
    #hyperparemeters and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)
    
    #data preprocessing   
    mean = [0.51825233, 0.50104613, 0.41361094]
    std = [0.26689603, 0.26242321, 0.27955858]
    train_set = AnimalDataset(root = args.data_path,
                              mean = mean,
                              std = std,
                              train = True,
                              size = args.image_size)
    train_loader = DataLoader(train_set,
                              batch_size = args.batch_size,
                              shuffle = True,
                              num_workers = 8,
                              drop_last = True,)
    eval_set = AnimalDataset(root = args.data_path,
                             mean = mean,
                             std = std,
                             train = False,
                             size = args.image_size)
    eval_loader = DataLoader(eval_set,
                             batch_size = args.batch_size,
                             shuffle = False,
                             num_workers = 8,
                             drop_last = False)
    
    #modeling
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    #edit the last layer of the model
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(train_set.categories), bias = True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    if args.resume_training:
        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        start_epoch = saved_data["epoch"]
        best_accuracy = saved_data["best_accuracy"]
    else:
        start_epoch = 0
        best_accuracy = 0
   
    #training phase
    for epoch in range(start_epoch, args.num_epochs):
        total_losses = 0
        progress_bar = tqdm(train_loader, colour = "green")
        model.train()
        for iter, (images, labels) in enumerate(progress_bar):
            #forward pass
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            #calculate loss
            loss = criterion(output, labels)
            total_losses += loss.item()
            avg_loss = total_losses / (iter + 1)
            progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}, Device: {device}")
            writer.add_scalar("Train/Loss",
                              avg_loss, 
                              epoch * len(train_loader) + iter)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        #evaluation phase
        model.eval()
        total_losses = 0
        y_true = []
        y_pred = []
        progress_bar = tqdm(eval_loader, colour = "yellow")

        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)

                total_losses += criterion(output, labels).item()
                y_true.extend(labels.tolist())
                predictions = output.argmax(dim = 1)
                y_pred.extend(predictions.tolist())

        accuracy = accuracy_score(y_true, y_pred)
        loss = total_losses / len(eval_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        writer.add_scalar("Eval/Accuracy", accuracy, epoch)
        plot_confusion_matrix(writer, confusion_matrix(y_true, y_pred), train_set.categories, epoch)

        #save the latest model (for resume training)
        saved_data = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_accuracy": max(accuracy, best_accuracy),
            "mean": mean,
            "std": std,
            "categories": train_set.categories
        }
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        torch.save(saved_data, checkpoint)

        #save the best model(for deployment)
        if accuracy > best_accuracy:
            checkpoint = os.path.join(args.checkpoint_path, "best.pt")
            torch.save(saved_data, checkpoint)
            best_accuracy = accuracy

if __name__ == "__main__":
    args = get_args()
    train(args)