import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from source import load_data, feature_extraction, protocol, classification, load_feature
import pickle
import yaml
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
#%%

def yaml_loader(file_path):
    with open(file_path, "r") as file_descriptor:
        info_data = yaml.safe_load(file_descriptor)
    return info_data


#%% load yaml

filepath = 'Config1.yaml'
info_data = yaml_loader(filepath)

# %%Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# Hyper-parameters
num_epochs = 20
batch_size = 10
learning_rate = 0.001


#%%
paths = {0: "save\\x_train", 1: "save\\y_train"}

train_data = load_feature.FeatureData(paths)

paths = {0: "save\\x_test", 1: "save\\y_test"}

test_data = load_feature.FeatureData(paths)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


#%% Class

architecture = info_data.get("Architecture")

class ConvNet(nn.Module):
    def __init__(self,architecture):
        super(ConvNet, self).__init__()

        l1 = 0
        l2 = 0
        l3 = 0
        for i1, k1 in enumerate(architecture.keys()):
            level = architecture[k1]
            element = '('
            for i2, k2 in enumerate(level.keys()):
               element = element + str(k2)+'='+str(level[k2])+', '
            element = element[0:-2] + ')'

            if k1[0:4] == 'conv':
                element = 'nn.Conv2d' + element
                l1 += 1
                exec('self.conv' + str(l1) + ' = ' + element)
            elif k1[0:4] == 'pool':
                element = 'nn.MaxPool2d' + element
                l2 += 1
                exec('self.pool' + str(l2) + ' = ' + element)
            elif k1[0:6] == 'linear':
                l3 += 1
                element = 'nn.Linear' + element
                exec('self.fc' + str(l3) + ' = ' + element)

            self.drp = nn.Dropout(p=0.2)



    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool1(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool2(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = self.pool3(F.relu(self.conv3(x)))  # -> n, 16, 5, 5
        x = x.view(-1,  x.shape[1] * x.shape[2] * x.shape[3])  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = self.drp(x)
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x


#%%

model = ConvNet(architecture).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

loss_g = []

for epoch in range(num_epochs):
    print(epoch)
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images.float())
        labels = labels.type(torch.LongTensor)

        loss = criterion(outputs, np.squeeze(labels))
        loss_g.append(loss)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2 == 0:

            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

#%%

all_pred = np.array([])
all_lab = np.array([])
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(28)]
    n_class_samples = [0 for i in range(28)]
    for images, labels in test_loader:
        if  len(all_lab) != 0:
            all_lab =np.concatenate((all_lab, torch.Tensor.numpy(labels)))
        else:
            all_lab = torch.Tensor.numpy(labels)


        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images.float())
        labels = labels.type(torch.LongTensor)
        _, predicted = torch.max(outputs, 1)

        if  len(all_pred) != 0:
            all_pred = np.concatenate((all_pred, torch.Tensor.numpy(predicted)))
        else:
            all_pred = torch.Tensor.numpy(predicted)

        n_samples += labels.size(0)
        n_correct += (predicted == labels.T).sum().item()


    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

all_pred = all_pred.reshape((len(all_pred), 1))


conf = confusion_matrix(all_lab, all_pred)

sm = conf.sum(axis = 1)
cp = sm.reshape((len(sm), 1))*np.ones((1, len(sm)))


df_cm = pd.DataFrame(conf/cp, range(27), range(27))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size

plt.show()