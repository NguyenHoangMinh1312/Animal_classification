"""
Training my own CNN model. You can use other avalable models (Resnet, VGG, ...) instead.
"""
from dataset import AnimalDataset
from model import CNN
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#This returns the argument to change the hyperparameters of the model
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", "-d", type = str, default = "./datasets/animals", help = "Path to the dataset")
    parser.add_argument("--learning_rate", "-l", type = float, default = 1e-3, help = "Learning rate of the optimizer")
    parser.add_argument("--batch_size", "-b", type = int, default = 4, help = "Number of images that goes into models each time")
    parser.add_argument("--epochs", "-e", type = int,  default = 100, help = "Number of epochs")
    parser.add_argument("--checkpoint_path", "-c", type = str, default = "./animal_classification/checkpoints", help = "checkpoint folder")
    parser.add_argument("--tensorboard_path", "-t", type = str, default = "./animal_classification/tensorboard", help = "tensorboard folder")
    parser.add_argument("--resume_training", "-r", type = bool, default = True, help = "Continue training from previous model or not")

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
    #hyperparemeters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    
    #data preprocessing
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    train_set = AnimalDataset(root = args.data_path,
                              mean = mean,
                              std = std,
                              train = True)
    train_dataloader = DataLoader(dataset = train_set,
                                  batch_size = args.batch_size,
                                  shuffle = True,
                                  drop_last = True,
                                  num_workers = 4)
    val_set = AnimalDataset(root = args.data_path,
                            mean = mean,
                            std = std,
                            train = False)
    val_dataloader = DataLoader(dataset = val_set,
                                batch_size = args.batch_size,
                                shuffle = False,
                                drop_last = False,
                                num_workers = 4)
    
    #model, loss function and optimizer
    model = CNN(num_classes = len(train_set.classes))
    model.to(device)
    optimizer = Adam(params = model.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.resume_training:
        model_path = os.path.join(args.checkpoint_path, "last.pt")
        saved_data = torch.load(model_path)

        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        start_epoch = saved_data["epoch"]
        best_accuracy = saved_data["best_accuracy"]
    else:
        start_epoch = 0
        best_accuracy = 0   #this value is to compare accuracy between models
    
    for epoch_id in range(start_epoch, args.epochs):
        #trainng phase
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader)
        for batch_id, (image, label) in enumerate(progress_bar):
            #forward pass
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            #calculate loss
            loss = criterion(output, label)
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch:{epoch_id + 1}/{args.epochs}, avg_loss: {total_loss/(batch_id + 1)}, device: {device}")
            writer.add_scalar("Train/Loss", total_loss/(batch_id + 1), epoch_id * len(train_dataloader) + batch_id)
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #validation phase
        model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        progress_bar = tqdm(val_dataloader)
        with torch.no_grad():
            for batch_id, (image, label) in enumerate(progress_bar):
                #forward pass
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                predictions = torch.argmax(output, dim = 1)
                
                #calculate loss
                loss = criterion(output, label)
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch:{epoch_id + 1}/{args.epochs}, avg_loss: {total_loss/(batch_id + 1)}, device: {device}")
                writer.add_scalar("Eval/Loss", total_loss/(batch_id + 1), epoch_id * len(val_dataloader) + batch_id)

                y_true.extend(label.tolist())
                y_pred.extend(predictions.tolist())
            accuracy = accuracy_score(y_true, y_pred)
            writer.add_scalar("Eval/Accuracy", accuracy, epoch_id)
            plot_confusion_matrix(writer, confusion_matrix(y_true, y_pred), train_set.classes, epoch_id)
            
        #save the model
        saved_data= {"epoch": epoch_id + 1,
                     "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "best_accuracy": max(accuracy, best_accuracy),
                     "mean": mean,
                     "std": std,
                     "classes": train_set.classes}
        last_model_path = os.path.join(args.checkpoint_path, "last.pt")
        torch.save(saved_data, last_model_path)

        if accuracy > best_accuracy:
            best_model_path = os.path.join(args.checkpoint_path, "best.pt")
            torch.save(saved_data, best_model_path)
            best_accuracy = accuracy

if __name__ == "__main__":
    args = get_args()
    train(args)

        










    

    
    

