import torchvision.models as models
import torch.nn as nn
import torch        
        
class Seq_Model(nn.Module):
    def __init__(self,num_classes,embed_dim=512,device='cuda',aggregation_head='attn'):
        super().__init__()
        self.feature_extractor=nn.Sequential(nn.Conv2d(1,64,7,2,3),*(list(models.resnet18().children())[1:-1]))
        self.aggregation_head=aggregation_head
        
        temp=torch.empty((1,embed_dim))
        nn.init.xavier_uniform_(temp)
        self.cls_token=nn.Parameter(temp.to(device))
        
        encoder_layer=nn.TransformerEncoderLayer(embed_dim,8,2*embed_dim,0.2,'relu',batch_first=True)
        self.aggregation_head=nn.TransformerEncoder(encoder_layer,4)
        
        self.classification_head=nn.Linear(embed_dim,num_classes)
    
    def forward(self,x:torch.Tensor,src_key_padding_mask:torch.Tensor)->torch.Tensor:
        # x.shape=(batch,n_scans,channels,height,width)
        x=torch.permute(x,(0,2,1,3,4))
        b,num_scan,c,h,w=x.shape
        
        x=x.view(-1,c,h,w)
        features=self.feature_extractor(x)
        features=features.view(b,num_scan,-1)
        
        features=torch.concat([self.cls_token.unsqueeze(0).expand(b,-1,-1),features],dim=1)
        assert features.shape[0:2]==(b,num_scan+1),'the logic implemented is not right'
        

        output=self.aggregation_head(features,src_key_padding_mask=src_key_padding_mask)
        # print(output.shape)
        
        
        assert features.shape==output.shape, 'unexcepted result'
        
        outputs=output[:,0,:].squeeze(1)
        # print(outputs.shape)
        logits=self.classification_head(outputs)
        
        return logits
class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(nn.ConvTranspose2d)
class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: modify for 1-channel input
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # [B, 64, H/2, W/2]
            *list(models.resnet18().children())[1:-2],  # exclude avgpool and fc
        )
        
        # Decoder: upsampling path to reconstruct
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [B, 256, H/32, W/32]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B, 128, H/16, W/16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # [B, 64, H/8, W/8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # [B, 32, H/4, W/4]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # [B, 16, H/2, W/2]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),     # [B, 1, H, W]
            nn.Sigmoid()  # use Tanh or None for other ranges
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded       

if __name__=="__main__":
    from torchsummary import summary
    
    model=Seq_Model(5,device='cpu')
    # model(torch.ones(2,200,1,256,256))
    summary(model,(200,1,256,256),2)
    
    '''
    torch summary actually does not work due to a nuance
    '''
        
        
        
        
        
        
        