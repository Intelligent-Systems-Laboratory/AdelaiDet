import torch

model = torch.load('EN-B3.pth', map_location ='cpu')
print(model)