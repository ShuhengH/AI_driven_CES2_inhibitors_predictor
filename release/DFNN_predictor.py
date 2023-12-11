import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# create dataset for DNN
class RegressionDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

# convert dataset to tensor
def data_process(data, test_size, random_state):
    x = data.iloc[:,2:].to_numpy()
    y = data["pIC50"].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=random_state)
    
    train_dataset = RegressionDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test_dataset = RegressionDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
                                     
    return x_train,x_test,y_train,y_test,train_dataset,test_dataset

# create NN model class
#Class MultipleRegression(nn.Module):
class Net(nn.Module):
    def __init__(self, num_features):
        #super(MultipleRegression, self).__init__()
        super(Net, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_4 = nn.Linear(16, 16)
        #self.layer_5 = nn.Linear(16, 16)
        self.layer_out = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        #x = self.relu(self.layer_5(x))
        x = self.layer_out(x)
        
        return (x)

    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        #x = self.relu(self.layer_5(x))
        x = self.layer_out(x)

        return (x)

# training DFNN model
def fit_NN_model(train_loader,test_loader,EPOCHS,BATCH_SIZE,LEARNING_RATE,NUM_FEATURES):

    model = Net(NUM_FEATURES)
    
    # check if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #print("DEVICE:",device,"MODEL Parameters:",model)
    
    criterion = nn.MSELoss(size_average=None)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optional optimizer: SGD, ASGD, Rprop, Adagrad, Adadelta, RMSprop, Adam(AMSGrad), Adamax, SparseAdam, LBFGS
    
    loss_status = {"train": [],
                   "test": []}
    
    print("Begin training.")
    
    for e in tqdm(range(1, EPOCHS+1)):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            
        
        # VALIDATION
        with torch.no_grad():
        
            test_epoch_loss = 0
            model.eval()
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            
                y_test_pred = model(X_test_batch)
                test_loss = criterion(y_test_pred, y_test_batch.unsqueeze(1))
            
                test_epoch_loss += test_loss.item()
            loss_status['train'].append(train_epoch_loss/len(train_loader))
            loss_status['test'].append(test_epoch_loss/len(test_loader))
    
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Test Loss: {test_epoch_loss/len(test_loader):.5f}')
    
    #y_pred_train = model(torch.from_numpy(x_train).float()).tolist()
    #y_pred_test = model(torch.from_numpy(x_test).float()).tolist()
    
    return loss_status,model

# predicting validation set
def NN_predict(data, path, num_features):
    x = data.iloc[:,2:].to_numpy()
    
    model = Net(num_features)
    model.load_state_dict(torch.load(path))
    #model = torch.load(path)
    y_pred_validation = model(torch.from_numpy(x).float()).tolist()
    
    pre_result = pd.DataFrame(y_pred_validation)
    pre_result.columns=list(["Pred_pIC50"])
    #pre_result.to_csv("pIC50_for_validation_set.csv", index = True)
    
    return pre_result
