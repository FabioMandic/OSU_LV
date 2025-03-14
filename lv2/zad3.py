import numpy as np
import matplotlib . pyplot as plt

image = plt.imread('road.jpg')
plt.imshow(image, cmap='gray', alpha=0.25)
plt.title('Prosvjetljena slika')
plt.show()

plt.imshow(image[:,image.shape[1]//4 : image.shape[1]//2], cmap='gray')
plt.title('Druga cetvrtina po sirini')
plt.show()

image_rot = np.rot90(image)
plt.imshow(image_rot,cmap='gray')
plt.title('Rotirana slika')
plt.show()

image_flip = np.fliplr(image)
plt.imshow(image_flip,cmap='gray')
plt.title('Zrcalna slika')
plt.show()