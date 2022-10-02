import streamlit as st 
from models import resnetModel

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset, random_split
import PIL
from PIL import Image


resnetModel.load_state_dict(torch.load('animals-resnet.pth', map_location=torch.device('cpu')))
resnetModel.eval()

stTransforms = tt.Compose([tt.Resize((128,128)),
                        tt.RandomCrop(96, padding=4, padding_mode='reflect'),
                        tt.ToTensor(), 
                        #tt.Normalize(*colour_stats)
                        ])

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

classes = ['cat','dog','panda']
  
st.title("CAT-DOG-PANDA Classifier")
st.write("")
st.caption("Using Residual Neural Networks to detect if the image contains a 🙀, 🐶, or 🐼")
st.write("https://github.com/ahmedsadid/CAT_DOG_PANDA")

st.subheader("Upload an image below to try it out!")
st.caption("Model will run automatically.")
file_up = st.file_uploader("", type="jpg")

st.subheader("Result")
if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    outIdx = np.argmax(resnetModel(image_loader(stTransforms, file_up)).detach().numpy())
    outputText = "You have uploaded a " + classes[outIdx] + "!"
    st.markdown(outputText)
