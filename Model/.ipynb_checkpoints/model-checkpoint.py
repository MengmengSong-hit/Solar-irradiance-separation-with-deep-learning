import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Attention import *
from Model.Transformer import *

    class CAT(nn.Module):
    def __init__(self,h_size,cnn_dim1,cnn_dim2,k1,k2,rat,device,atti):
        super(CAT,self).__init__()
        self.h_size = h_size
        self.cnn_dim1 = cnn_dim1
        self.cnn_dim2 = cnn_dim2
        self.k1=k1
        self.k2=k2
        self.rat=rat
        
        self.CNN1=nn.Conv2d(1, self.cnn_dim1, (1,self.k1), stride=1, bias=True)
        self._activation1=nn.ReLU()
        
        if atti==1:  
            
            self.attention1=channel_attention1(self.cnn_dim1, ratio= rat)
        elif atti==2: 
            
            self.attention1=channel_attention2(self.cnn_dim1, ratio= rat)
        elif atti==3:
            
            self.attention1=channel_attention3(self.cnn_dim1, ratio= rat)
        elif atti==4: 
            
            self.attention1=channel_attention4(self.cnn_dim1, ratio= rat)
        elif atti==5: 
            
            self.attention1=channel_attention5(self.cnn_dim1, ratio= rat)
        elif atti==6: 
            
            self.attention1=channel_attention6(self.cnn_dim1, ratio= rat)
        elif atti==7: 
            
            self.attention1=channel_attention7(self.cnn_dim1, ratio= rat)
        elif atti==8: 
            
            self.attention1=channel_attention8(self.cnn_dim1, ratio= rat)
        elif atti==9: 
            
            self.attention1=channel_attention9(self.cnn_dim1, ratio= rat)
        else : 
            
            self.attention1=channel_attention10(self.cnn_dim1, ratio= rat)
        
        self.CNN2=nn.Conv2d(self.cnn_dim1, self.cnn_dim2,(1,self.k2), stride=1, bias=True)
        self._activation2=nn.ReLU()
        
        if atti==1:   
            self.attention2=channel_attention1(self.cnn_dim1, ratio= rat)
        elif atti==2: 
            self.attention2=channel_attention2(self.cnn_dim1, ratio= rat)
        elif atti==3: 
            self.attention2=channel_attention3(self.cnn_dim1, ratio= rat)
        elif atti==4: 
            self.attention2=channel_attention4(self.cnn_dim1, ratio= rat)
        elif atti==5: 
            self.attention2=channel_attention5(self.cnn_dim1, ratio= rat)
        elif atti==6: 
            self.attention2=channel_attention6(self.cnn_dim1, ratio= rat)
        elif atti==7: 
            self.attention2=channel_attention7(self.cnn_dim1, ratio= rat)
        elif atti==8: 
            self.attention2=channel_attention8(self.cnn_dim1, ratio= rat)
        elif atti==9: 
            self.attention2=channel_attention9(self.cnn_dim1, ratio= rat)
        else : 
            self.attention2=channel_attention10(self.cnn_dim1, ratio= rat)
            
        self.CNN3=nn.Conv2d(self.cnn_dim2,1,(1,1), stride=1, bias=True)
        self._activation3=nn.ReLU()
        
        self._dense = nn.Linear((self.h_size+2-self.k1-self.k2),2)
        
    def forward(self, h):
        h=h.unsqueeze(1) 
        h=h.unsqueeze(1) 
        h1=self.CNN1(h)
        h1=self._activation1(h1)
        h1=self.attention1(h1)
        h2=self.CNN2(h1)
        h2=self._activation2(h2)
        h2=self.attention2(h2)
        h3=self.CNN3(h2)
        h3=self._activation3(h3)
        h3 = h3.flatten(1)
        forecast = self._dense (h3)
        
        summing=forecast [:,0].add(forecast[:,1])
        kd_ratio=forecast[:,0].div(summing)
        kb_ratio=forecast[:,1].div(summing)
        kd_ratio=kd_ratio.unsqueeze(-1)
        kb_ratio=kb_ratio.unsqueeze(-1)
        return kd_ratio,kb_ratio

class FC(nn.Module):
    def __init__(self,input_size,h_size,device=None):
        super(FC, self).__init__()
        self._input_size = input_size
        self._h_size = h_size
        self._device = device
        self.fc=nn.Linear(self._input_size,self._h_size)
        self._activation = torch.sigmoid
        
    def forward(self, x):
        hidden = self.fc(x)
        hidden = self._activation(hidden)
        return hidden
    
class seperate_modeling(nn.Module):
    def __init__(self,input_size,elm_h_size,cnn_dim1,cnn_dim2,k1,k2,rat,device,atti,ffn_hidden,n_head,n_layers):
        super(seperate_modeling,self).__init__( )
        self.BlockList = nn.ModuleList([CAT(elm_h_size,cnn_dim1,cnn_dim2,k1,k2,rat,device,atti)])
        self.BlockList.extend([CAT(elm_h_size,cnn_dim1,cnn_dim2,k1,k2,rat,device,atti) for _ in range(input_size-1)])
        self.Transformer=Transformer(elm_h_size,ffn_hidden,n_head, n_layers,input_size,device)
        #self.weight_raw = torch.nn.Parameter(torch.zeros(input_size)).to(device)   
        #self._activation = torch.sigmoid
        self.dense = nn.ModuleList([FC(1,elm_h_size)])
        self.dense.extend([FC( _+2,elm_h_size) for _ in range(input_size-1)])
        
        
    def forward(self, input_data):
        batch_size,num_of_features=input_data.shape 
        new_input=[]
        #self.weight_raw=self._activation(self.weight_raw)
        for i in range (num_of_features):
            #selected_col=self.weight_raw.topk(k=i+1, dim=0, largest=True)[1]
            #print(selected_col)
            selected_col=torch.arange(0,i+1,1)
            tmp=self.dense[i](input_data[:,selected_col])
            new_input.append(tmp.unsqueeze(1))
        new_input=torch.cat(new_input,1)  
        
        kd_ratio_list=[]
        kb_ratio_list=[]
        kd_ratio,kb_ratio,x=self.Transformer(new_input)
        kd_ratio_list.append(kd_ratio)  
        kb_ratio_list.append(kb_ratio) 
        
        for f_index in range(num_of_features):
            elm_feature=new_input[:,f_index,:]
            kd_ratio,kb_ratio=self.BlockList[f_index](elm_feature)
            kd_ratio_list.append(kd_ratio)  
            kb_ratio_list.append(kb_ratio)    
        
        kd_ratio_ensem=torch.cat(kd_ratio_list,1)
        kb_ratio_ensem=torch.cat(kb_ratio_list,1)
        kd_ratio_ensem =kd_ratio_ensem.unsqueeze(1)
        kb_ratio_ensem =kb_ratio_ensem.unsqueeze(1)
        
        return kd_ratio_ensem,kb_ratio_ensem,x

class get_weight(nn.Module):
    def __init__(self,elm_h_size,input_size):
        super(get_weight,self).__init__( )
        self.eps = 1e-7
        self.weight_raw = nn.Linear(elm_h_size,input_size+1)
        #self._activation=nn.abs()
        
    def forward(self, input_data):
        weight=self.weight_raw(input_data)
        #weight=self._activation(weight)
        weight=torch.abs(weight)
        weight = weight.div(weight.sum(dim=1).clamp(min=self.eps).unsqueeze(-1))
        weight =weight.unsqueeze(-1)
        return weight
        
class Proposed_model(nn.Module):
    def __init__(self,input_size,elm_h_size,cnn_dim1,cnn_dim2,k1,k2,rat,device,atti,ffn_hidden,n_head,n_layers):
        super(Proposed_model, self).__init__()
        self.seperate_modeling=seperate_modeling(input_size,elm_h_size,cnn_dim1,cnn_dim2,k1,k2,rat,device,atti,ffn_hidden,n_head,n_layers)
        self.weight=get_weight(ffn_hidden*15,input_size)
        self.weight1= nn.Linear(input_size+1,1)
        self.weight2= nn.Linear(input_size+1,1)
        
    def forward(self, input_data):
        kd_ratio_ensem,kb_ratio_ensem,x=self.seperate_modeling(input_data)
        weight=self.weight(x)
        
        kd_ratio_result=kd_ratio_ensem.bmm(weight).squeeze(-1)
        kb_ratio_result=kb_ratio_ensem.bmm(weight).squeeze(-1)
        
        out=torch.cat((kd_ratio_result,kb_ratio_result),dim=1) 
        return out 
    
