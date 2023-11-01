import cv2
import os
import matplotlib.pyplot as plt

class DifferencesCalculate:
  def display(input_folder, result_folder):
    dir = os.getcwd()
    img_input = DifferencesCalculate.imread(dir + "\\" + input_folder)
    img_output = DifferencesCalculate.imread(dir + "\\" + result_folder)
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1) 
    plt.title('Original Image (In)', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('High Quality (Out)', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img_input)
    ax2.imshow(img_output)
    fig.savefig("differences_" + img_output)
  
  def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img