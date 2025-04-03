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
        

if __name__=="__main__":
    from torchsummary import summary
    
    model=Seq_Model(5,device='cpu')
    # model(torch.ones(2,200,1,256,256))
    summary(model,(200,1,256,256),2)
    
    '''
    torch summary actually does not work due to a nuance
    '''
        
        
        
        
        
        
        