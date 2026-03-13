import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class channel_attention1(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.shared_MLP=nn.Sequential(
            #nn.Conv2d(channel, channel // ratio,1, bias=False),
            #nn.ReLU(),
            #nn.Conv2d(channel // ratio, channel, 1, bias=False)
        #)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel,channel),
        #nn.ReLU()
         )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.avg_pool(x).squeeze()
        maxout = self.max_pool(x).squeeze()
        weight = self.sigmoid(avgout + maxout)
        weight=self.shared_MLP(weight)
        #weight=weight.unsqueeze(-1) 
        #weight=weight.unsqueeze(-1) 
        #avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        #maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight= self.sigmoid(avgout + maxout)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention2(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.shared_MLP=nn.Sequential(
            #nn.Conv2d(channel, channel // ratio,1, bias=False),
            #nn.ReLU(),
            #nn.Conv2d(channel // ratio, channel, 1, bias=False)
        #)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel,channel),
        #nn.ReLU()
         )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight = self.sigmoid(avgout + maxout)
        #weight=weight.unsqueeze(-1) 
        #weight=weight.unsqueeze(-1) 
        #avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        #maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention3(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.shared_MLP=nn.Sequential(
            #nn.Conv2d(channel, channel // ratio,1, bias=False),
            #nn.ReLU(),
            #nn.Conv2d(channel // ratio, channel, 1, bias=False)
        #)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel,channel),
        nn.ReLU()
         )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight = self.sigmoid(avgout + maxout)
        #weight=weight.unsqueeze(-1) 
        #weight=weight.unsqueeze(-1) 
        #avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        #maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention4(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention4, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel,channel),
        nn.ReLU()
         )

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight = avgout + maxout
        #weight=weight.unsqueeze(-1) 
        #weight=weight.unsqueeze(-1) 
        #avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        #maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention5(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention5, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel,channel),
        nn.Sigmoid()
         )

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight = avgout + maxout
        #weight=weight.unsqueeze(-1) 
        #weight=weight.unsqueeze(-1) 
        #avgout = self.shared_MLP(self.avg_pool(x).squeeze())
        #maxout = self.shared_MLP(self.max_pool(x).squeeze())
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention6(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention6, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel*2,channel),
        nn.Sigmoid()
         )

    def forward(self, x):
        avgout = self.avg_pool(x).squeeze()
        maxout = self.max_pool(x).squeeze()
        weight = torch.cat((avgout,maxout),dim=1) 
        weight = self.shared_MLP(weight)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention7(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention7, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel*2,channel),
        nn.ReLU()
         )

    def forward(self, x):
        avgout = self.avg_pool(x).squeeze()
        maxout = self.max_pool(x).squeeze()
        weight = torch.cat((avgout,maxout),dim=1) 
        weight = self.shared_MLP(weight)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention8(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention8, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel*2,channel),
        nn.Softmax()
         )

    def forward(self, x):
        avgout = self.avg_pool(x).squeeze()
        maxout = self.max_pool(x).squeeze()
        weight = torch.cat((avgout,maxout),dim=1) 
        weight = self.shared_MLP(weight)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention8(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention8, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP=nn.Sequential(
        nn.Linear(channel*2,channel),
        nn.Softmax()
         )

    def forward(self, x):
        avgout = self.avg_pool(x).squeeze()
        maxout = self.max_pool(x).squeeze()
        weight = torch.cat((avgout,maxout),dim=1) 
        weight = self.shared_MLP(weight)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention9(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention9, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.MLP1=nn.Sequential(
        nn.Linear(channel,channel),
        nn.ReLU()
         )
        self.MLP2=nn.Sequential(
        nn.Linear(channel,channel),
        nn.ReLU()
         )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.MLP1(self.avg_pool(x).squeeze())
        maxout = self.MLP2(self.max_pool(x).squeeze())
        weight = self.sigmoid(avgout + maxout)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out
    
class channel_attention10(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention10, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.MLP=nn.Sequential(
        nn.Linear(channel,channel),
        nn.ReLU()
         )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.avg_pool(x).squeeze()
        maxout = self.max_pool(x).squeeze()
        weight = self.sigmoid(avgout + maxout)
        weight = self.MLP (weight)
        weight=weight.unsqueeze(-1) 
        weight=weight.unsqueeze(-1) 
        out=x*weight
        return out