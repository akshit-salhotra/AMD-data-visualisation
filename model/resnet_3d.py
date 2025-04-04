import torch.nn as nn
import torch

class BasicConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel,downsample=True):
        super().__init__()
        layer=[]
        self.downsample=downsample
        if self.downsample:
            layer.append(BasicConvBlock.create_conv(in_ch,out_ch,kernel,2,kernel//2))
            self.conv1x1=nn.Conv3d(in_ch,out_ch,1,2)
        else:
            layer.append(BasicConvBlock.create_conv(out_ch,out_ch,kernel,1,kernel//2))

        for i in range(3):
            layer.append(BasicConvBlock.create_conv(out_ch,out_ch,kernel,1,kernel//2))
        # layer.append(BasicConvBlock.create_conv(out_ch,out_ch,kernel,2,kernel//2))
        self.layers=nn.Sequential(*layer)
        

    @staticmethod
    def create_conv(in_ch,out_ch,kernel,stride,padding):
        return(nn.Sequential(nn.Conv3d(in_ch,out_ch,kernel,stride,padding),nn.BatchNorm3d(out_ch),nn.ReLU()))
    
    def forward(self,x):
        if self.downsample:
            x_conv1x1=self.conv1x1(x)
            
        for i in range(2):
            x1=self.layers[i](x)
            x2=self.layers[i+1][0](x1)
            if i==0 and self.downsample:
                x=self.layers[i+1][1:](x_conv1x1+x2)
            else:
                x=self.layers[i+1][1:](x+x2)
        return x
        
        
class Resnet18_3D(nn.Module):
    
    def __init__(self,num_classes):
        super().__init__()
        self.conv1=Resnet18_3D.create_conv(1,64,7,2,3)
        self.maxpool=nn.MaxPool3d(3,2,padding=1)
        self.conv2=BasicConvBlock(64,64,3,downsample=False)
        self.conv3=BasicConvBlock(64,128,3)
        self.conv4=BasicConvBlock(128,256,3)
        self.conv5=BasicConvBlock(256,512,3)
        self.avgPool=nn.AdaptiveAvgPool3d(1)
        self.linear=nn.Linear(512,num_classes)
        

    @staticmethod
    def create_conv(in_ch,out_ch,kernel,stride,padding):
        return(nn.Sequential(nn.Conv3d(in_ch,out_ch,kernel,stride,padding),nn.BatchNorm3d(out_ch),nn.ReLU()))
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avgPool(x)
        x=torch.flatten(x,start_dim=1)
        x=self.linear(x)
        
        return x
        
    
if __name__=='__main__':
    from torchsummary import summary
    Device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Resnet18_3D(6).to(Device)
    summary(model,(1,128,256,256),1)
    # print(list(model.children()))