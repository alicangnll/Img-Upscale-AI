import torch
from PIL import Image
from RealESRGAN import *

input = 'demo.png'
output = 'output.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device)
model.load_weights('model/RealESRGAN_x4plus.pth')
image = Image.open(input).convert('RGB')
sr_image = model.predict(image)
sr_image.save(output)
DifferencesCalculate.display(input, output)
