import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    # input pre-processing
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # get MNIST datasets
    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False,
                              transform=custom_transform)
    # return data loader accordingly to the parameter training
    if training:
        return torch.utils.data.DataLoader(train_set, batch_size=50)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size=50)


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),           # a Flatten layer to convert the 2D pixel array to a 1D array
        nn.Linear(784, 128),    # a Dense layer with 128 nodes
        nn.ReLU(),
        nn.Linear(128, 64),     # a Dense layer with 64 nodes
        nn.ReLU(),
        nn.Linear(64, 10),      # a Dense layer with 10 nodes
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()       # set model to training mode
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)     # setup optimizer
    for epoch in range(T):      # loop over the dataset T times
        train_loss = 0.0        # average loss per epoch (accumulated loss in an epoch / length of the dataset)
        correct = 0             # number of correctly predicted label for each epoch
        for i, data in enumerate(train_loader, 0):      # 60000 / 50 = 1200 batches in total
            inputs, labels = data       # get inputs to NN and labels from train_loader
            opt.zero_grad()             # clear old gradients
            # forward, get loss, backward, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            # accumulate train_loss for each batch in an epoch iteration
            train_loss += loss.item() * train_loader.batch_size
            # count number of correct prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        # print training status for each epoch
        print("​Train Epoch: %d   Accuracy: %d/60000(%.2f%%)   Loss: %.3f" %
              (epoch, correct, correct / len(train_loader.dataset) * 100, train_loss / len(train_loader.dataset)))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()        # set model to evaluation mode
    with torch.no_grad():
        eva_loss = 0.0          # evaluation loss
        correct = 0             # number of correctly predicted label for each epoch
        for data, labels in test_loader:
            outputs = model(data)       # get NN outputs
            loss = criterion(outputs, labels)       # get loss
            eva_loss += loss.item() * test_loader.batch_size        # accumulate evaluation loss for each batch
            # count number of correct prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        # display evaluation results accordingly to show_loss
        if show_loss:
            print("Average loss: %.4f" % (eva_loss / len(test_loader.dataset)))
        print("Accuracy: %.2f%%" % (correct / len(test_loader.dataset) * 100))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    test_image = test_images[index]         # get the test image
    output = model(test_image)              # get output from NN
    prob = F.softmax(output, dim=1)         # apply softmax to output
    # sort probabilities
    prob = prob.tolist()[0]
    indexes = np.argsort(prob)
    # display the top three most likely class labels
    print(class_names[indexes[9]], ": %.2f%%" % (prob[indexes[9]] * 100))
    print(class_names[indexes[8]], ": %.2f%%" % (prob[indexes[8]] * 100))
    print(class_names[indexes[7]], ": %.2f%%" % (prob[indexes[7]] * 100))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
