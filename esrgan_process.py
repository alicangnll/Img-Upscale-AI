import torch
from PIL import Image
from RealESRGAN import *

input_pic = 'pic\\demo.png'
output_pic = 'pic\\output.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4, anime=False)
# If anime is true : model.load_weights('model/RealESRGAN_x4plus_anime_6B.pth')
model.load_weights('model/RealESRGAN_x4plus.pth')
image = Image.open(input_pic).convert('RGB')
sr_image = model.predict(image)
sr_image.save(output_pic)
DifferencesCalculate.display(input_pic, output_pic)
RealESRGAN.face_enchange(output_pic)