import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn,optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class HealthIndexDataset(Dataset):

    def __init__(self,csv_file,transform=None):

        self.health_frame = pd.read_csv(csv_file)
        self.features = ["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]

    def __len__(self):
        return self.health_frame.shape[0]
    
    def __getitem__(self, index):
        row = self.health_frame.iloc[index]
        Y = torch.tensor(row['Score'],dtype=torch.float32)
        X = torch.tensor(row[self.features],dtype=torch.float32)
        return X,Y
    
class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        yhat = self.linear(x)
        return yhat
    


dataset = HealthIndexDataset('./2018.csv')
# indices = list(range(len(dataset)))
# split = (int(np.floor(0.2*len(dataset))))
# np.random.seed(1)
# np.random.shuffle(indices)

# train_indices, test_indices = indices[split:], indices[:split]
# print(f'len train: {len(train_indices)}')
# print(f'len test: {len(test_indices)}')

# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)

health_data_train = DataLoader(dataset=dataset,batch_size=5)
# health_data_test = DataLoader(dataset=dataset,batch_size=1,sampler=test_sampler)

lr = 0.01
# lr = 0.005
epochs = 20

model = LR(6,1)
optimizer = optim.SGD(model.parameters(),lr)
criterion = nn.MSELoss()

LOSS = []

print(list(model.parameters()))
print(dataset[0])

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    total = 0

    for X,Y in health_data_train:
        # print(X)
        # print(Y)
        Yhat = model(X)
        loss = criterion(Yhat.view(-1,),Y)
        total += loss.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    LOSS.append(total/5)

plt.plot(LOSS)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.show()

health_data_test = HealthIndexDataset('./2019.csv')
health_data_test_loader = DataLoader(dataset=health_data_test,batch_size=1)

LOSS2 = []
for X,Y in health_data_test_loader:
    # print(X,Y)
    Yhat = model(X)
    loss2 = criterion(Yhat.view(-1,),Y)

    LOSS2.append(loss2.item())

plt.plot(LOSS2)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.show()


