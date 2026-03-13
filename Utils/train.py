import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Model.model import *
import os
import copy
  
def train(epochs, model, train_dataset,val_dataset,optimizer,criterion,device,bs):
    train_input= torch.tensor(train_dataset[:,0:(train_dataset.shape[1]-2)], dtype=torch.float32)
    train_label= torch.tensor(train_dataset[:,[15,16,17]], dtype=torch.float32)
    train_data= Data.TensorDataset(train_input, train_label)
    train_loader= Data.DataLoader(dataset=train_data,batch_size=bs,shuffle=True)
    
    val_input= torch.tensor(val_dataset[:,0:(val_dataset.shape[1]-2)], dtype=torch.float32)
    val_label = torch.tensor(val_dataset[:,[15,16,17]], dtype=torch.float32)
    val_data = Data.TensorDataset(val_input, val_label)
    val_loader = Data.DataLoader(dataset=val_data,batch_size=bs,shuffle=True)
    
    train_loss_all=[]
    val_loss_all=[]
    for epoch in range(epochs):
        #print(f"############## Epoch:{epoch} ##############")
        model.train()
        epoch_loss = 0
        num=0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i, (data, lable) in loop:
            data=data.to(device)
            lable= lable.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*data.shape[0]
            num+=data.shape[0]
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=epoch_loss/num)     
        train_loss_all.append(epoch_loss /num)
        validation_loss=compute_val_loss(model,val_loader,criterion,device)
        val_loss_all.append(validation_loss)
    plt.figure()
    plt.plot(train_loss_all,"r.-",label = "Train loss")
    plt.plot(val_loss_all,"b.-",label = "Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return val_loss_all[-1]

def train_elman(epochs, model, train_dataset,val_dataset,optimizer,criterion,device,bs,ElmanModel):
    train_input=ElmanModel.output(train_dataset[:,0:(train_dataset.shape[1]-2)])
    train_input= torch.tensor(train_input,dtype=torch.float32)
    train_label= torch.tensor(train_dataset[:,[15,16,17]], dtype=torch.float32)
    train_data= Data.TensorDataset(train_input, train_label)
    train_loader= Data.DataLoader(dataset=train_data,batch_size=bs,shuffle=True)
    
    val_input=ElmanModel.output(val_dataset[:,0:(val_dataset.shape[1]-2)])
    val_input= torch.tensor(val_input, dtype=torch.float32)
    val_label = torch.tensor(val_dataset[:,[15,16,17]], dtype=torch.float32)
    val_data = Data.TensorDataset(val_input, val_label)
    val_loader = Data.DataLoader(dataset=val_data,batch_size=bs,shuffle=True)
    
    train_loss_all=[]
    val_loss_all=[]
    for epoch in range(epochs):
        #print(f"############## Epoch:{epoch} ##############")
        model.train()
        epoch_loss = 0
        num=0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i, (data, lable) in loop:
            data=data.to(device)
            lable= lable.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*data.shape[0]
            num+=data.shape[0]
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=epoch_loss/num)     
        train_loss_all.append(epoch_loss /num)
        validation_loss=compute_val_loss(model,val_loader,criterion,device)
        val_loss_all.append(validation_loss)
    plt.figure()
    plt.plot(train_loss_all,"r.-",label = "Train loss")
    plt.plot(val_loss_all,"b.-",label = "Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return val_loss_all[-1]

def masked_input(input_data,ElmanModel1,ElmanModel2):
    all_input=[]
    all_input.append(ElmanModel2.output(input_data[:,np.newaxis,:]))
    for i in range(input_data.shape[1]):
        mask_data=copy.copy(input_data)
        select=mask_data[:,i][:,np.newaxis]
        mask_data=ElmanModel1.output(select)
        mask_data=mask_data[:,np.newaxis,:]
        all_input.append(mask_data)
    all_input=np.concatenate(all_input,axis=1)
    return all_input

def train_proposed(epochs, model, train_dataset,val_dataset,optimizer,criterion,device,bs,hid,filename):
    train_input= torch.tensor(train_dataset[:,0:(train_dataset.shape[1]-2)],dtype=torch.float32)
    
    train_label= torch.tensor(train_dataset[:,[14,15,16]], dtype=torch.float32)
    train_data= Data.TensorDataset(train_input, train_label)
    train_loader= Data.DataLoader(dataset=train_data,batch_size=bs,shuffle=True)
    
    val_input= torch.tensor(val_dataset[:,0:(val_dataset.shape[1]-2)], dtype=torch.float32)
    
    val_label = torch.tensor(val_dataset[:,[14,15,16]], dtype=torch.float32)
    val_data = Data.TensorDataset(val_input, val_label)
    val_loader = Data.DataLoader(dataset=val_data,batch_size=bs,shuffle=True)
    
    train_loss_all=[]
    val_loss_all=[]
    for epoch in range(epochs):
        #print(f"############## Epoch:{epoch} ##############")
        model.train()
        epoch_loss = 0
        num=0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i, (data, lable) in loop:
            data=data.to(device)
            lable= lable.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*data.shape[0]
            num+=data.shape[0]
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=epoch_loss/num)     
        train_loss_all.append(epoch_loss /num)
        validation_loss=compute_val_loss(model,val_loader,criterion,device)
        val_loss_all.append(validation_loss)
        if (epoch+1)%100==0:
            torch.save(model.state_dict(), f'Model_saving/Proposed_{filename}_hidden{hid}_ep{epoch+1}.pt')
    plt.figure()
    plt.plot(train_loss_all,"r.-",label = "Train loss")
    plt.plot(val_loss_all,"b.-",label = "Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    min_ep=np.argmin(val_loss_all)+1
    print(f"Min validation loss epoch {min_ep}")
    return val_loss_all[-1]

def train_autoencoder_part1(epochs, model, train_dataset,val_dataset,optimizer,criterion,device,bs):
    train_input= torch.tensor(train_dataset[:,0:(train_dataset.shape[1]-2)], dtype=torch.float32)
    train_label= torch.tensor(train_dataset[:,0:(train_dataset.shape[1]-2)], dtype=torch.float32)
    train_data= Data.TensorDataset(train_input, train_label)
    train_loader= Data.DataLoader(dataset=train_data,batch_size=bs,shuffle=True)
    val_input= torch.tensor(val_dataset[:,0:(val_dataset.shape[1]-2)], dtype=torch.float32)
    val_label= torch.tensor(val_dataset[:,0:(val_dataset.shape[1]-2)], dtype=torch.float32)
    val_data= Data.TensorDataset(val_input, val_label)
    val_loader= Data.DataLoader(dataset=val_data,batch_size=bs,shuffle=True)
    
    train_loss_all=[]
    val_loss_all=[]
    for epoch in range(epochs):
        #print(f"############## Epoch:{epoch} ##############")
        model.train()
        epoch_loss = 0
        num=0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i, (data, lable) in loop:
            data=data.to(device)
            lable= lable.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*data.shape[0]
            num+=data.shape[0]
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=epoch_loss/num)     
        train_loss_all.append(epoch_loss /num)
        validation_loss=compute_val_loss(model,val_loader,criterion,device)
        val_loss_all.append(validation_loss)
    plt.figure()
    plt.plot(train_loss_all,"r.-",label = "Train loss")
    plt.plot(val_loss_all,"b.-",label = "Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return val_loss_all[-1]

def train_autoencoder_part2(epochs, model, train_dataset,val_dataset,optimizer,criterion,device,bs):
    train_input= torch.tensor(train_dataset[:,0:(train_dataset.shape[1]-3)], dtype=torch.float32)
    train_label= torch.tensor(train_dataset[:,(train_dataset.shape[1]-3):train_dataset.shape[1]], dtype=torch.float32)
    train_data= Data.TensorDataset(train_input, train_label)
    train_loader= Data.DataLoader(dataset=train_data,batch_size=bs,shuffle=True)
    
    val_input= torch.tensor(val_dataset[:,0:(val_dataset.shape[1]-3)], dtype=torch.float32)
    val_label = torch.tensor(val_dataset[:,(val_dataset.shape[1]-3):val_dataset.shape[1]], dtype=torch.float32)
    val_data = Data.TensorDataset(val_input, val_label)
    val_loader = Data.DataLoader(dataset=val_data,batch_size=bs,shuffle=True)
    
    train_loss_all=[]
    val_loss_all=[]
    for epoch in range(epochs):
        #print(f"############## Epoch:{epoch} ##############")
        model.train()
        epoch_loss = 0
        num=0
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i, (data, lable) in loop:
            data=data.to(device)
            lable= lable.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*data.shape[0]
            num+=data.shape[0]
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=epoch_loss/num)     
        train_loss_all.append(epoch_loss /num)
        validation_loss=compute_val_loss(model,val_loader,criterion,device)
        val_loss_all.append(validation_loss)
    plt.figure()
    plt.plot(train_loss_all,"r.-",label = "Train loss")
    plt.plot(val_loss_all,"b.-",label = "Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return val_loss_all[-1]

def compute_val_loss(net,val_loader,criterion,device):
    net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        epoch_loss = 0
        num=0
        for batch_index, (inputs, labels) in enumerate(val_loader):
            inputs=inputs.to(device)
            labels= labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_loss +=loss.item()*inputs.shape[0]
            num+=inputs.shape[0]
        validation_loss = epoch_loss /num
    return validation_loss

def predict(model,device,test_dataset):
    print(f"test_dataset{test_dataset.shape}")
    test_X= torch.tensor(test_dataset[:,0:(test_dataset.shape[1]-2)], dtype=torch.float32)
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=10000,shuffle=False)
    result=np.array([0,0])
    model.eval()
    for _data in test_loader:
        data = _data[0].to(device)
        output = model(data).detach().cpu().numpy()
        result=np.vstack((result,output))
    result=result[1:,] 
    return result    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

def predict_proposed(model,device,test_dataset):
    test_X= torch.tensor(test_dataset[:,0:(test_dataset.shape[1]-2)],dtype=torch.float32)
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=10000,shuffle=False)
    result=np.array([0,0])
    model.eval()
    for _data in test_loader:
        data = _data[0].to(device)
        output = model(data).detach().cpu().numpy()
        result=np.vstack((result,output))
    result=result[1:,] 
    return result    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

def predict_autoencoder(model,device,test_dataset):
    test_X= torch.tensor(test_dataset[:,0:(test_dataset.shape[1]-3)], dtype=torch.float32)
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=test_X.shape[0],shuffle=False)
    # 预测过程
    model.eval()
    for _data in test_loader:
        data = _data[0].to(device)
        output = model(data)
    #print(f"output{output.shape}")
    return output.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

def predict_elman(model,device,test_dataset,ElmanModel):
    test_X= ElmanModel.output(test_dataset[:,0:(test_dataset.shape[1]-2)])
    test_X= torch.tensor(test_X,dtype=torch.float32)
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=test_X.shape[0],shuffle=False)
    # 预测过程
    model.eval()
    for _data in test_loader:
        data = _data[0].to(device)
        output = model(data)
    #print(f"output{output.shape}")
    return output.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

def predict_elman(model,device,test_dataset,ElmanModel):
    test_X= ElmanModel.output(test_dataset[:,0:(test_dataset.shape[1]-2)])
    test_X= torch.tensor(test_X,dtype=torch.float32)
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=test_X.shape[0],shuffle=False)
    # 预测过程
    model.eval()
    for _data in test_loader:
        data = _data[0].to(device)
        output = model(data)
    #print(f"output{output.shape}")
    return output.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据

class My_loss_way1(nn.Module):#由0.08下降到0.04
    def __init__(self):
        super().__init__()
        
    def forward(self,forecast, y):
        summing=forecast [:,0].add(forecast[:,1])
        kd_ratio=forecast[:,0].div(summing)
        kb_ratio=forecast[:,1].div(summing)
        kd=kd_ratio.mul(y[:,0])
        kb=kb_ratio.mul(y[:,0])
        return torch.mean(torch.pow((kd - y[:,1]), 2))+torch.mean(torch.pow((kb- y[:,2]), 2))
    
def convert_way1(forecast,kt):
    print(forecast[:,0].shape)
    summing=forecast[:,0]+forecast[:,1]
    kd_ratio=forecast[:,0]/summing
    kb_ratio=forecast[:,1]/summing
    forecast[:,0]=kt*kd_ratio
    forecast[:,1]=kt*kb_ratio
    return forecast
    #kd=forecast.flatten()
    #kb=kt.flatten()-kd
    #forecast=np.concatenate((kd[:,np.newaxis],kb[:,np.newaxis]),axis=1)
    
class My_loss_way2(nn.Module)
    def __init__(self):
        super().__init__()
        
    def forward(self,forecast, y):
        kd=(y[:,0].mul(forecast[:,0].sub(forecast[:,1])+1))*0.5    
        kb=(y[:,0].mul(forecast[:,1].sub(forecast[:,0])+1))*0.5
        return torch.mean(torch.pow((kd - y[:,1]), 2))+torch.mean(torch.pow((kb- y[:,2]), 2))

def convert_way2(forecast,kt):
    kd=kt*(forecast[:,0]-forecast[:,1]+1)*0.5
    kb=kt*(forecast[:,1]-forecast[:,0]+1)*0.5
    forecast=np.concatenate((kd[:,np.newaxis],kb[:,np.newaxis]),axis=1)
    return forecast 

class proposed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,forecast, y):
        kd=y[:,0].mul(forecast[:,0])  
        kb=y[:,0].mul(forecast[:,1])
        return torch.mean(torch.pow((kd - y[:,1]), 2))+torch.mean(torch.pow((kb- y[:,2]), 2))
    
def proposed_convert(forecast,kt):
        kd=kt*forecast[:,0] 
        kb=kt*forecast[:,1]
        forecast=np.concatenate((kd[:,np.newaxis],kb[:,np.newaxis]),axis=1)
        return forecast
    
def seed_torch(seed):
    random.seed(seed)#设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  