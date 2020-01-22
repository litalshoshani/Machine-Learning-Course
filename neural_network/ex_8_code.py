#lital shoshani
#204071427
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

class Model1(nn.Module):
    def __init__(self,image_size):
        super(Model1, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #x is turning into a vector
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim = 1)


class Model2(nn.Module):
    def __init__(self,image_size):
        super(Model2, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #x is turning into a vector
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim = 1)

class Model3(nn.Module):
    def __init__(self,image_size):
        super(Model3, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, x):
        #x is turning into a vector
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn1(self.fc0(x)))
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x,dim = 1)


def train(optimizer, model, train_loader,size):
    model.train()
    correct = 0
    train_loss = 0;
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        #do forward - compute the network output
        output = model(data)
        #the next two lines a friend helped me write, in order to make my training better
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        # compute loss
        loss = F.nll_loss(output, labels, size_average=False)
        train_loss += loss
        #do backward
        loss.backward()
        optimizer.step()

    train_loss /= size
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, size,
        100. * correct / size))

    return float(train_loss)


def test(test_loader,model, split_size):
    model.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= split_size
    print('\nValiation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, split_size,
        100. * correct / split_size))
    return test_loss



def write_pred(model, test_loader,size):
    model.eval()
    test_loss = 0
    correct = 0
    s = ""
    for data,target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        s += str(pred[0][0].item())+ "\n"
    test_pred_file = open("test.pred", "w")
    test_pred_file.write(s)
    test_pred_file.close()
    test_loss /= size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, size,
        100. * correct / size))




#in this method I used: https://am207.github.io/2018spring/wiki/ValidationSplits.html
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)

    test_dataset = datasets.FashionMNIST(root='./data',train=False,transform=transform)

    # Define the indices
    indices = list(range(len(train_dataset)))
    #get the size of the 80% of the training set
    split = int(0.2*len(train_dataset))

    batch_size = 64

    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1, sampler=validation_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    #model = Model1(image_size=28 * 28)
    #model = Model2(image_size=28 * 28)
    model = Model3(image_size=28 * 28)

    #optimizer = optim.SGD(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.AdaDelta(model.parameters(), lr=lr)
    #optimizer = optim.RMSprop(model.parameters(), lr=lr)
    test_x = torch.from_numpy(np.loadtxt("test_x")).float()

    train_loss_dict = {}
    validation_loss_dict = {}

    for epoch in range(1, 10 + 1):
        print('\nEpoch: ' + str(epoch) + '\n')

        train_loss = train(optimizer,model,train_loader,len(train_loader)*batch_size)
        validatioin_loss = test(validation_loader,model,split)

        train_loss_dict[epoch] = train_loss
        validation_loss_dict[epoch] = validatioin_loss


    train_loss_list = sorted(train_loss_dict.items())
    x,y = zip(*train_loss_list)
    plt.plot(x, y, "r-", label='train')

    validation_loss_list = sorted(validation_loss_dict.items())
    z,w = zip(*validation_loss_list)
    plt.plot(z , w, "b-", label='validation')
    plt.legend()
    plt.show()

    write_pred(model, test_loader,len(test_loader))

if __name__ == '__main__':
    main()



