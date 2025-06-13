import torch
from utils.util import SliceLevelPerceptualLoss
from memory_profiler import profile

@profile
def work():
    perceptual_device=torch.device('cuda')

    perceptual_loss_fn=SliceLevelPerceptualLoss()
    recons_scans=torch.rand(1,1,128,256,256).to(perceptual_device)
    scans=torch.rand(1,1,128,256,256).to(perceptual_device)

    # with torch.cuda.device(perceptual_device):
    recons_scans = recons_scans.to(perceptual_device)
    scans = scans.to(perceptual_device)
    p_loss = perceptual_loss_fn(recons_scans.reshape(-1,1,256,256), scans.reshape(-1,1,256,256))
    print(p_loss)

work()
print(torch.cuda.memory_summary())