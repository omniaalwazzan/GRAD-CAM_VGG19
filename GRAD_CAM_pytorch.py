### This a re-implementation of the code presented under this link: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82  

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

#%%

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    #%%
path = r"C:\Users\Omnia\Pictures\data/" # this suppose to contain folder of classes /dog/cat
path_to_save_heatmap = r"C:\Users\Omnia\Pictures\data\dog\map.jpg"
path_to_single_img = r"C:\Users\Omnia\Pictures\data\dog\dog.jpg"

transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root=path, transform=transform)
# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
#%%


vgg = VGG()
vgg.eval()
# get the image from the dataloader
img, _ = next(iter(dataloader))
# get the most likely prediction of the model
pred = vgg(img)
index= pred.argmax(dim=1)
tensor_value = index[0].item()
pred[:, tensor_value].backward()
#%%

# pull the gradients out of the model
gradients = vgg.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = vgg.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

heatmap /= torch.max(heatmap)

#plt.matshow(heatmap.squeeze())
heatmap_np = heatmap.detach().cpu().numpy()

#%%

img = cv2.imread(path_to_single_img)
heatmap = cv2.resize(heatmap_np, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
#cv2.imwrite(path_to_save_heatmap, superimposed_img)

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax2.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
ax2.set_title('Grad-CAM Heatmap')
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax2)
cbar.set_label('Intensity')
plt.show()
