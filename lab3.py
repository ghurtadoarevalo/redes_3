from skimage import io
from matplotlib import pyplot as plt
import numpy as np
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
kernel2 = np.array([[1/256, 4/256, 6/256, 4.0/256 ,1.0/256],[4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],[6.0/256, 24.0/256, 36.0/256, 24.0/256 , 6.0/256],[4.0/256, 16.0/256 ,24.0/256, 16.0/256 ,4.0/256],[1/256, 4/256, 6/256, 4.0/256 ,1.0/256]])
img = io.imread('leena512.bmp',as_gray=True)  # load the image as grayscale


output = np.zeros_like(img)


image_padded = np.zeros((img.shape[0]+2, img.shape[1]+2))

print("SHAPE 0 ",img.shape[0])
print("SHAPE 0 ",img.shape[1])

image_padded[1:-1, 1:-1] = img

for x in range(img.shape[1]):     # Loop over every pixel of the image
        for y in range(img.shape[0]):
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()

print(image_padded)
print(output)

plt.imshow(kernel, interpolation='nearest',cmap='gray')
plt.show()

plt.imshow(kernel2, interpolation='nearest',cmap='gray')
plt.show()
