# Helper function for extracting features from pre-trained models
import torch
from torchvision import transforms

import matplotlib.pyplot as plt

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_tensor, backbone, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):


    img_tensor = transforms.functional.resize(img_tensor, 112)
    # extract features
    backbone.to(device).eval() # set to evaluation mode
    with torch.no_grad():
            features = l2_norm(backbone(img_tensor))
            
#     np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features


if __name__ == "__main__":
    None