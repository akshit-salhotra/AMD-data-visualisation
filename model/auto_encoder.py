from model.resnet_3d import BasicConvBlock
import torch.nn as nn
import torch
import torch.nn.functional as F

class Interpolate(nn.Module):
    def forward(self,x):
        return F.interpolate(x,scale_factor=2.0,mode='trilinear')

        
class BasicUpsampleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel):
        super().__init__()
        self.upsample=nn.ConvTranspose3d(in_ch,out_ch,kernel,stride=2,padding=1,output_padding=1)
        self.norm=nn.BatchNorm3d(out_ch)
        self.relu=nn.ReLU()

        self.conv=nn.Sequential(*[BasicConvBlock.create_conv(out_ch,out_ch,kernel,1,kernel//2) for _ in range(3)])
        
        self.resize_conv=nn.ConvTranspose3d(in_ch,out_ch,2,2)

    def forward(self,x):
        out=self.relu(self.norm(self.upsample(x)))
        out=self.conv[0][0](out)+self.resize_conv(x)

        out=self.conv[0][1:](out)

        out=self.conv[1](out)
        out=self.conv[2](out)

        return out

class AutoEncoder(nn.Module):
    def __init__(self,ch=[64,128,256,512]):
        super().__init__()

        self.conv1=BasicConvBlock.create_conv(1,ch[0],7,2,3)
        self.maxpool=nn.MaxPool3d(3,2,padding=1)
        self.conv2=BasicConvBlock(ch[0],ch[0],3,downsample=False)
        self.conv3=BasicConvBlock(ch[0],ch[1],3)
        self.conv4=BasicConvBlock(ch[1],ch[2],3)
        self.conv5=BasicConvBlock(ch[2],ch[3],3)

        self.encoder=nn.Sequential(self.conv1,self.maxpool,self.conv2,self.conv3,self.conv4,self.conv5)

        self.decode2=BasicUpsampleBlock(ch[0],ch[0],3)
        self.decode3=BasicUpsampleBlock(ch[1],ch[0],3)
        self.decode4=BasicUpsampleBlock(ch[2],ch[1],3)
        self.decode5=BasicUpsampleBlock(ch[3],ch[2],3)

        self.final_decoder=nn.Conv3d(ch[0],1,7,1,3)

        self.decoder=nn.Sequential(self.decode5,
                                   self.decode4,
                                   self.decode3,
                                   self.decode2,
                                   Interpolate(),
                                   self.final_decoder
                                   )
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.encoder(x)
        out=self.decoder(x)
        return self.sigmoid(out)

    @staticmethod
    def get_decoder(ch):
        pass

    @staticmethod
    def get_encoder(ch):
        pass


if __name__=="__main__":
    from torchsummary import summary

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model=AutoEncoder().to(device)
    # this data is misleading due to repetitions
    # summary(model,(1,128,256,256),2)

    print(model(torch.ones((1,1,128,256,256)).to(device)).shape)
    # print(model)
