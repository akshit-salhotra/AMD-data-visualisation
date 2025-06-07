import torch
import logging
import torch.nn as nn
import torchvision.models as models

def load_model(args,model):
    params=torch.load(args.model_path)['state_dict']

    # for key in params.keys():
    #      if key.startswith('module.'):
    #           params[f'module.{key}']=params.pop(key)
        
    missing,unexpected=model.load_state_dict(params,strict=False)
    logging.info(f'the missing keys are : \n{missing}')
    logging.info(f'the unexpected keys are :\n{unexpected}')
    print('loaded model parameters from ',args.model_path)
    logging.info(f'loaded model parameters from {args.model_path}')
    
    
class SliceLevelPerceptualLoss(nn.Module):
    def __init__(self, layer='relu3_3'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.blocks = nn.Sequential()

        layer_map = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23
        }

        self.blocks = vgg[:layer_map[layer] + 1].eval()
        for param in self.blocks.parameters():
            param.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, x, y):
        '''
        x and y be bscan level entities
        '''
        assert x.device==y.device, f' the device of x and y is not same'
        # Normalize inputs to match VGG expectations
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        self.blocks.to(x.device)
        # Compute perceptual features
        fx = self.blocks(x)
        fy = self.blocks(y)

        return nn.functional.mse_loss(fx, fy)
    
    def get_feature_map(self,x):

        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        self.blocks.to(x.device)
        fx = self.blocks(x)
        
        print(fx.shape)
        
        return fx
