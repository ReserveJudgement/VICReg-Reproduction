import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import faiss
import matplotlib.pyplot as plt
import pickle

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1).to(device)
        self.resnet.maxpool = nn.Identity().to(device)
        self.resnet.fc = nn.Linear(512, 512).to(device)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D)).to(device)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   ).to('cuda')

    def forward(self, x):
        return self.model(x)


def get_CIFAR10():
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def train_vic(trainset, testset, D=128, batch_size=256, lamda=25., mu=25., nu=1., epochs=30, name="VICReg", knn=None):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    f = Encoder()
    h = Projector(D=D)
    model = nn.Sequential(f, h)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-04, betas=(0.9, 0.999), weight_decay=1e-06)
    inv_loss = nn.MSELoss()
    relu = nn.ReLU()
    losses = {"invariance loss": [], "variance loss": [], "covariance loss": []}
    test_epoch = {"invariance loss": [], "variance loss": [], "covariance loss": []}
    for _ in tqdm(range(epochs)):
        for x, _ in trainloader:
            model.train()
            # augment to get two views
            original = []
            augmentation = []
            for pic in range(x.size()[0]):
                original.append(torch.unsqueeze(x[pic, :, :, :], dim=0))
                if knn is None:
                    augmentation.append(torch.unsqueeze(train_transform(x[pic, :, :, :]), dim=0))
                else:
                    neighb = knn[pic, random.randint(0, 2)].item()
                    augmentation.append(torch.unsqueeze(trainset[neighb][0], dim=0))
            x_a = torch.cat(original).to('cuda')
            x_b = torch.cat(augmentation).to('cuda')
            # get representations
            z_a = model(x_a)
            z_b = model(x_b)
            # calculate invariance loss
            invariance_loss = inv_loss(z_a, z_b)
            losses['invariance loss'].append(invariance_loss.cpu().detach().item())
            # calculate variance loss
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            variance_loss = torch.mean(relu(1 - std_z_a)) + torch.mean(relu(1 - std_z_b))
            losses['variance loss'].append(variance_loss.cpu().detach().item())
            # calculate covariance loss
            z_a_meandif = z_a - z_a.mean(dim=0)
            z_b_meandif = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a_meandif.T @ z_a_meandif) / (batch_size - 1)
            cov_z_b = (z_b_meandif.T @ z_b_meandif) / (batch_size - 1)
            covariance_loss = (cov_z_a.fill_diagonal_(0).pow(2).sum() + cov_z_b.fill_diagonal_(0).pow(2).sum()) / D
            losses['covariance loss'].append(covariance_loss.cpu().detach().item())
            # full loss
            loss = (lamda * invariance_loss) + (mu * variance_loss) + (nu * covariance_loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # get test
        test_inv = []
        test_var = []
        test_cov = []
        for y, _ in testloader:
            model.eval()
            original = []
            augmentation = []
            for pic in range(y.size()[0]):
                original.append(torch.unsqueeze(y[pic, :, :, :], dim=0))
                augmentation.append(torch.unsqueeze(train_transform(y[pic]), dim=0))
            y_a = torch.cat(original, dim=0).to('cuda')
            y_b = torch.cat(augmentation, dim=0).to('cuda')
            z_a = model(y_a)
            z_b = model(y_b)
            invariance_loss = inv_loss(z_a, z_b)
            test_inv.append(invariance_loss.cpu().detach().item())
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            variance_loss = torch.mean(relu(1 - std_z_a)) + torch.mean(relu(1 - std_z_b))
            test_var.append(variance_loss.cpu().detach().item())
            z_a_meandif = z_a - z_a.mean(dim=0)
            z_b_meandif = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a_meandif.T @ z_a_meandif) / (batch_size - 1)
            cov_z_b = (z_b_meandif.T @ z_b_meandif) / (batch_size - 1)
            covariance_loss = (cov_z_a.fill_diagonal_(0).pow(2).sum() + cov_z_b.fill_diagonal_(0).pow(2).sum()) / D
            test_cov.append(covariance_loss.cpu().detach().item())
        test_epoch["invariance loss"].append(sum(test_inv)/len(test_inv))
        test_epoch["variance loss"].append(sum(test_var)/len(test_var))
        test_epoch["covariance loss"].append(sum(test_cov)/len(test_cov))
    # save
    state_dict = f.state_dict()
    torch.save(state_dict, f"{name}.pt")
    with open('trainloss.pkl', 'wb') as f:
        pickle.dump(losses, f)
    with open('testloss.pkl', 'wb') as f:
        pickle.dump(test_epoch, f)
    return losses, test_epoch


def plot_losses(train, test):
    fig, ax = plt.subplots(1, 3)
    for idx, loss in enumerate(["invariance loss", "variance loss", "covariance loss"]):
        ax[idx].set_title(loss)
        ax[idx].plot([x for x in range(len(train[loss]))], train[loss], c='blue', label="train")
        stride = int(len(train[loss])/len(test[loss]))
        ax[idx].plot([stride * x for x in range(len(test[loss]))], test[loss], c='red', label="test")
        ax[idx].legend(loc="upper right")
    plt.show()
    return


def Visualize2D(model, data, embedder="TSNE"):
    loader = DataLoader(data, batch_size=100, shuffle=False, num_workers=1)
    if embedder == "PCA":
        projector = PCA(n_components=2)
    elif embedder == "TSNE":
        projector = TSNE(n_components=2)
    colors = []
    colorkey = {1: 'tab:blue',
                2: 'tab:orange',
                3: 'tab:green',
                4: 'tab:red',
                5: 'tab:purple',
                6: 'tab:brown',
                7: 'tab:pink',
                8: 'tab:gray',
                9: 'tab:olive',
                0: 'tab:cyan'}
    reps = []
    for pic, cls in tqdm(loader):
        for c in cls:
            colors.append(colorkey[c.item()])
        processed = []
        for p in pic:
            processed.append(torch.unsqueeze(test_transform(p).to('cuda'), dim=0))
        processed = torch.vstack(processed)
        reps.append(model(processed).detach())
    X = torch.vstack(reps).detach().cpu().numpy()
    canvas = projector.fit_transform(X)
    plt.scatter(canvas[:, 0], canvas[:, 1], c=colors)
    plt.show()
    return


def LinearProbe(model, trainset, testset, batch_size):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Linear(128, 10, device='cuda')
    classifier.train()
    activation = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.requires_grad = True
    optim = torch.optim.Adam(classifier.parameters())
    print("")
    print("training linear probe")
    for _ in tqdm(range(10)):
        for data, label in trainloader:
            rep = model(data.to('cuda'))
            predict = activation(classifier(rep))
            true = F.one_hot(label, num_classes=10).type(torch.float).to('cuda')
            loss = loss_fn(predict, true)
            loss.backward()
            optim.step()
            optim.zero_grad()
    # evaluate
    print("")
    print("evaluating")
    classifier.eval()
    acc = 0
    for data, label in testloader:
        predict = torch.argmax(activation(classifier(model(data.to('cuda')))), dim=1)
        acc += int(torch.sum(torch.eq(predict, label.to('cuda'))).item())
    print("test accuracy: ", acc/(len(testset)))
    return acc


def get_representations(data, model, name=None):
    # function that receives a dataset and a model
    # returns the model's representation for the data
    # and saves it as a file if file name is defined
    # representations object is saved as torch tensor
    model.eval()
    reps = []
    # compute each representation and organize in tensor
    for image in tqdm(data):
        reps.append(model(torch.unsqueeze(test_transform(image), dim=0).cuda()).detach())
    reps = torch.vstack(reps)
    if name is not None:
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(reps, f)
    return reps


def FindNeighbors(data, query, k, name=None):
    # function to take dataset find n nearest neighbor for each image
    # returns array of pointer indices and saves file if name is defined
    reps = data.cpu().detach().numpy()
    query = query.cpu().detach().numpy()
    # use faiss to get knn
    index = faiss.IndexFlatL2(reps.shape[1])
    index.add(x=reps)
    densities, results = index.search(x=query, k=k+1)
    results = results[:, 1:]
    densities = densities[:, 1:]
    # take 1 random from each output if randomize parameter is set
    if name is not None:
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(results, f)
    return results, densities


def retrieval(data, neighbors):
    # function to receive dataset and table of n neighbors
    # it takes a random picture from each class, finds it's neighbors by the index table
    # and plots two rows at a time
    n = neighbors.shape[1]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for rows in range(5):
        fig, ax = plt.subplots(2, n + 1)
        for row, cls in enumerate([rows * 2, rows * 2 + 1]):
            imgs = [x for x in range(len(data)) if data[x][1] == cls]
            idx = random.choice(imgs)
            # plot knn results
            ax[row, 0].set_title(f"Reference: {classes[cls]}")
            ax[row, 0].imshow(transforms.ToPILImage()(data[idx][0]))
            for i in range(n):
                ax[row, i + 1].set_title(classes[data[int(neighbors[idx, i])][1]])
                ax[row, i + 1].imshow(transforms.ToPILImage()(data[int(neighbors[idx, i])][0]))
        plt.show()
    return


def cluster(data, labels, n):
    # function receives set of encoded images, their labels, and parameter n
    # it uses sklearn to make n clusters with k-means
    # then makes two tsne plots, one colored by clusters and the other colored by labels
    clusters = KMeans(n_clusters=n).fit(data)
    predict = clusters.predict(data).tolist()
    projector = TSNE(n_components=2)
    colorkey = {1: 'tab:blue',
                2: 'tab:orange',
                3: 'tab:green',
                4: 'tab:red',
                5: 'tab:purple',
                6: 'tab:brown',
                7: 'tab:pink',
                8: 'tab:gray',
                9: 'tab:olive',
                0: 'tab:cyan'}
    canvas = projector.fit_transform(data)
    centers = []
    for c in range(n):
        points = np.vstack([canvas[x, :] for x in range(len(predict)) if predict[x] == c])
        centers.append(np.mean(points, axis=0))
    centers = np.vstack(centers)
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Colored by Cluster')
    ax[0].scatter(canvas[:, 0], canvas[:, 1], c=[colorkey[x] for x in predict])
    ax[0].scatter(centers[:, 0], centers[:, 1], c='black')
    ax[1].set_title('Colored by Class')
    ax[1].scatter(canvas[:, 0], canvas[:, 1], c=[colorkey[x] for x in labels])
    ax[1].scatter(centers[:, 0], centers[:, 1], c='black')
    plt.show()
    return predict


if __name__ == '__main__':

    # Load model and get test set
    model = Encoder().to('cuda')
    loaded_dict = torch.load(f'VICReg.pt')
    model.load_state_dict(loaded_dict)
    train, test = get_CIFAR10()

    # Perform linear probing
    LinearProbe(model, train, test, 100)

    # Load embeddings
    images = train.data
    enc = get_representations(images, model)

    # Retrieve 5 nearest neighbors from random images in each class
    knn, _ = FindNeighbors(enc, enc, 5)
    retrieval(train, knn)

    # Clustering task
    labels = train.targets
    enc = enc.cpu().detach().numpy()
    predict = cluster(enc, labels, 10)
  
