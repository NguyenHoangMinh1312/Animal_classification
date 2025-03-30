import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2
from model import CNN

#This is the argument to change the hyperparameters of the model
def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--image-path", "-p", type = str, default = "animal_CNN/test_img/cat.jpg", help = "path to image")
    parser.add_argument("--model", "-m", type = str, default = "Animal_classification/checkpoints/best.pt", help = "path to model")
    args = parser.parse_args()
    return args

#This function are steps to train the model
def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_data = torch.load(args.model, weights_only = False)
    classes = saved_data["classes"]

    #Load the model
    model = CNN(num_classes = len(classes))
    model.load_state_dict(saved_data["model"])

    #preprocess the image
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    mean, std = saved_data["mean"], saved_data["std"]
    image = (image - mean) / std    
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()

    #inference
    model = model.to(device)
    model.eval()
    softmax = nn.Softmax()   
    with torch.no_grad():
        image = image.to(device)
        output = model(image)[0]
        predicted_class = classes[torch.argmax(output)]
        prob = softmax(output)[torch.argmax(output)]
    
    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (224, 224))
    cv2.imshow(f"{predicted_class}, Probability: {prob * 100:.2f}%", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    args = get_args()
    inference(args)

