import torch
import numpy as np
import torchvision.transforms as transforms
from zennit.attribution import Gradient
import zennit
from zennit.composites import Composite
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))

from src.glocal_analysis import run_analysis 
from src.datasets.flood_dataset import FloodDataset
from src.datasets.DLR_dataset import DatasetDLR
from src.plot_crp_explanations import plot_explanations, plot_one_image_explanation
from src.minio_client import MinIOClient
from LCRP.models import get_model 
from LCRP.utils.pidnet_canonizers import PIDNetBaseCanonizer, PIDNetCanonizer, EpsilonPlusFlatforPIDNet

# Define transformation (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])
# Load dataset
root_dir = "../datasets/data/General_Flood_v3/"
dataset = FloodDataset(root_dir=root_dir, split="train", transform=transform)

model_name = "pidnet"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading unet with path to checkpoint
model = get_model(model_name=model_name)


from LCRP.utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
import copy
from PIL import Image

def undo_to_tensor(tensor):
    images=[]

    for t in tensor:
        # Ensure tensor is detached from computation graph and moved to CPU
        t = t.detach().cpu()

        # Clamp to [0,1] in case of small floating point error
        t = torch.clamp(t, 0, 1)

        # Convert from [C, H, W] to [H, W, C] and scale to [0, 255]
        array = t.permute(1, 2, 0).numpy() * 255

        # Convert to uint8 for image representation
        array = array.astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(array)
        images.append(image)
    if len(images) == 1:
        images = images[0]
    return images
    
def LRP(x, model):
    with EpsilonPlusFlatforPIDNet().context(model) as modified_model:
        output = modified_model(x)
        # gradient/ relevance wrt. class/output 0
        init_grad=torch.zeros_like(output[1])
        init_grad[0,...]=1.
        output[1].backward(gradient=init_grad)
        attr=x.grad[0, 0, :, :].detach().cpu().numpy()
    return zennit.image.imgify(attr,
                               symmetric=True,)
N=5
plt.subplots(N,3, figsize=(10, 15))
for i in range(N):
    x = dataset[i][0].unsqueeze(0)
    model.eval()


    x.requires_grad_()
    # Without canonizer
    out_plain = model(x)[1]
    xpl=LRP(x, model)

    plt.subplot(N,3,3*i+1)
    if i==0:
        plt.title("Input")
    plt.imshow(undo_to_tensor(x))
    plt.axis('off')  
    plt.subplot(N,3,3*i+2)
    if i==0:
        plt.title("w.o. canonizer")
    plt.imshow(xpl)
    plt.axis('off')

    c=PIDNetCanonizer()
    h=c.apply(model)
    out_canon = model(x)[1]
    xpl=LRP(x, model)
    plt.subplot(N,3,3*i+3)
    if i==0:
        plt.title(f"w. canonizer")
    plt.imshow(xpl)
    plt.axis('off')  
    print(f"Output difference of canonizer after attaching: {(out_plain - out_canon).abs().max()}")
    for handle in h:
        handle.remove()
    out_canon = model(x)[1]
    print(f"Output difference of canonizer after detaching: {(out_plain - out_canon).abs().max()}")
    print("\n\n\n===\n\n")
plt.show()

#composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
#condition = [{"y": 1}]
#attr = attribution(copy.deepcopy(x).requires_grad_(), condition, composite, record_layer=["conv1.0"],
#                        init_rel=1)
#out_canon = attr.prediction
