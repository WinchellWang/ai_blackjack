import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):

    def __init__(self,input_size):
        super(NeuralNet,self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size,input_size*2,device=device)
        self.linear2 = nn.Linear(input_size*2,input_size*3,device=device)
        self.linear3 = nn.Linear(input_size*3,input_size*2,device=device)
        self.linear4 = nn.Linear(input_size*2,3,device=device)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return out

def model_training(x,y,num_epochs=50):
    x = x.to_numpy()
    y = y.to_numpy()
    neural_model = NeuralNet(input_size=8)
    optimizer  = torch.optim.SGD(neural_model.parameters(),lr=0.005)
    criterion = nn.L1Loss()
    loss_save = 99999
    y,x = torch.from_numpy(y.astype(np.float32)).to(device),torch.from_numpy(x.astype(np.float32)).to(device)
    for epoch in range(num_epochs):
        # forward path and loss
        y_pred = neural_model(x)
        train_loss = criterion(y_pred,y)
        # backward pass
        train_loss.backward()
        # update
        optimizer.step()
        # empty gradient
        optimizer.zero_grad()
        # if (epoch+1) % 100 == 0:
        #     print(f'[Neural Model]: epoch {epoch+1}, training_loss = {train_loss.item():.4f}')
        if loss_save > train_loss.item():
            neural_model_save = neural_model.state_dict().copy()
            loss_save = train_loss.item()   
    return neural_model_save

def model_prediction(x,neural_model):
    model = NeuralNet(input_size=8)
    model.load_state_dict(neural_model)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x.astype(np.float32)).to(device)
        y = model(x)
    return y.cpu().numpy()

def continue_training(model, x, y, num_epochs=200):
    x = x.to_numpy()
    y = y.to_numpy()
    neural_model = NeuralNet(input_size=8)
    neural_model.load_state_dict(model)
    optimizer = torch.optim.SGD(neural_model.parameters(), lr=0.005)
    criterion = nn.L1Loss()
    loss_save = 99999
    y,x = torch.from_numpy(y.astype(np.float32)).to(device),torch.from_numpy(x.astype(np.float32)).to(device)
    for epoch in range(num_epochs):
        # forward path and loss
        y_pred = neural_model(x)
        train_loss = criterion(y_pred,y)
        # backward pass
        train_loss.backward()
        # update
        optimizer.step()
        # empty gradient
        optimizer.zero_grad()
        # if (epoch+1) % 100 == 0:
        #     print(f'[Neural Model]: epoch {epoch+1}, training_loss = {train_loss.item():.4f}')
        if loss_save > train_loss.item():
            neural_model_save = neural_model.state_dict().copy()
            loss_save = train_loss.item()  
    return neural_model_save

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')