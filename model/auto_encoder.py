from model.resnet_3d import BasicConvBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

class Interpolate(nn.Module):
    def forward(self,x):
        return F.interpolate(x,scale_factor=2.0,mode='trilinear')

class Interpolate_2d(nn.Module):
    def forward(self,x):
        return F.interpolate(x,scale_factor=2.0,mode='bilinear')
        
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

        # self.conv1=BasicConvBlock.create_conv(1,ch[0],7,2,3)
        # self.maxpool=nn.MaxPool3d(3,2,padding=1)
        # self.conv2=BasicConvBlock(ch[0],ch[0],3,downsample=False)
        # self.conv3=BasicConvBlock(ch[0],ch[1],3)
        # self.conv4=BasicConvBlock(ch[1],ch[2],3)
        # self.conv5=BasicConvBlock(ch[2],ch[3],3)

        # self.encoder=nn.Sequential(self.conv1,self.maxpool,self.conv2,self.conv3,self.conv4,self.conv5)
        self.encoder=AutoEncoder.get_encoder(ch)

        # self.decode2=BasicUpsampleBlock(ch[0],ch[0],3)
        # self.decode3=BasicUpsampleBlock(ch[1],ch[0],3)
        # self.decode4=BasicUpsampleBlock(ch[2],ch[1],3)
        # self.decode5=BasicUpsampleBlock(ch[3],ch[2],3)

        # self.final_decoder=nn.Conv3d(ch[0],1,7,1,3)

        # self.decoder=nn.Sequential(self.decode5,
        #                            self.decode4,
        #                            self.decode3,
        #                            self.decode2,
        #                            Interpolate(),
        #                            self.final_decoder
        #                            )
        self.decoder=AutoEncoder.get_decoder(ch)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.encoder(x)
        out=self.decoder(x)
        return self.sigmoid(out)

    @staticmethod
    def get_decoder(ch):
        ch.reverse()
        l=len(ch)-1
        # ch.append(ch[-1])
        layers=[]
        # print('the number of upsample blocks are :',ch)
        for i in range(l):
            layers.append(BasicUpsampleBlock(ch[i],ch[i+1],3))
        
        layers.append(Interpolate())
        layers.append(nn.Conv3d(ch[-1],1,7,1,3))

        return nn.Sequential(*layers)

    @staticmethod
    def get_encoder(ch):
        layers=[]
        layers.append(BasicConvBlock.create_conv(1,ch[0],7,2,3))
        layers.append(nn.MaxPool3d(3,2,padding=1))
        l=len(ch)
        ch.insert(0,ch[0])

        for i in range(l):
            if i==0:
                layers.append(BasicConvBlock(ch[i],ch[i+1],3,downsample=False))
            else:
                layers.append(BasicConvBlock(ch[i],ch[i+1],3))
        
        return nn.Sequential(*layers)

class AutoEncoder_2d(nn.Module):

    def __init__(self,encoder_type):
        super().__init__()
        encoders={
            'resnet18':models.resnet18
            ,'resnet34':models.resnet34
        }
        aux_blocks={'resnet18':[0,0,0]
                    ,'resnet34':[8,4,2]}
        
        assert encoder_type in list(encoders.keys()) ,f"invalid encoder type"

        self.encoder=nn.Sequential(nn.Conv2d(1,64,7,2,3),*(list(encoders[encoder_type]().children())[1:-2]))
        self.decoder=AutoEncoder_2d.get_decoder(aux_blocks[encoder_type])
        self.sig=nn.Sigmoid()

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)

        return self.sig(x)

    @staticmethod
    def get_decoder(aux_blocks:list,ch=[64,128,256,512]):
        ch.reverse()
        l=len(ch)
        # ch.append(ch[-1])
        layers=[]
        # print('the number of upsample blocks are :',ch)
        for i in range(l-1):
            layers.append(Decoder_block_2d(ch[i],ch[i+1],3,aux_blocks[i]))
        
        layers.append(Interpolate_2d())
        layers.append(nn.ConvTranspose2d(ch[-1],ch[-1],7,2,3,1))
        layers.append(nn.Conv2d(ch[-1],1,1,1,0))

        return nn.Sequential(*layers)

class Decoder_block_2d(nn.Module):
    
    def __init__(self,in_ch,out_ch,kernel,aux_blocks=0):
        super().__init__()
        self.upsample=nn.ConvTranspose2d(in_ch,out_ch,kernel,stride=2,padding=1,output_padding=1)
        self.norm=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU()

        self.conv=nn.Sequential(*[Decoder_block_2d.create_conv(out_ch,out_ch,kernel,1,kernel//2) for _ in range(3)])
        # self.comv=nn.Sequential(*[Decoder_block_2d.create_conv(out_ch,out_ch,kernel,1,kernel//2) for _ in r])
        if aux_blocks>0:
            self.aux_conv=nn.Sequential(*[Decoder_block_2d.create_conv(out_ch,out_ch,kernel,1,kernel//2) for _ in range(aux_blocks)])
        self.aux_blocks=aux_blocks
        self.resize_conv=nn.ConvTranspose2d(in_ch,out_ch,2,2)

    
    def forward(self,x):
        out=self.relu(self.norm(self.upsample(x)))
        out=self.conv[0][0](out)+self.resize_conv(x)

        out=self.conv[0][1:](out)

        out=self.conv[1](out)
        out=self.conv[2](out)

        if self.aux_blocks>0:
            out=self.aux_conv(out)

        return out
    
    @staticmethod
    def create_conv(in_ch,out_ch,kernel,stride,padding):
        return(nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel,stride,padding),
                             nn.BatchNorm2d(out_ch,momentum=0.1),
                             nn.ReLU()))
    
if __name__=="__main__":
    from torchsummary import summary

    device=torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    
    model=AutoEncoder_2d('resnet34').to(device)
    # this data is misleading due to repetitions
    summary(model,(1,256,256),2,device='cpu')

    # print(model(torch.ones((1,1,128,256,256)).to(device)).shape)
    # print(model)

    # print(AutoEncoder.get_decoder([64,128,256,512]))
