from skimage import io
from matplotlib import pyplot as plt
import numpy as np
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
kernel2 = np.array(
    [
        [1.0/256, 4.0/256, 6.0/256, 4.0/256 ,1.0/256],
        [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
        [6.0/256, 24.0/256, 36.0/256, 24.0/256 , 6.0/256],
        [4.0/256, 16.0/256 ,24.0/256, 16.0/256 ,4.0/256],
        [1.0/256, 4.0/256, 6.0/256, 4.0/256 ,1.0/256]
    ]
)

img = io.imread('leena512.bmp',as_gray=True)  # load the image as grayscale

output = np.zeros_like(img)

largo = len(kernel2)

image_padded = np.zeros((img.shape[0]+largo-1, img.shape[1]+largo-1))

print("SHAPE 0 ",img.shape[0])
print("SHAPE 0 ",img.shape[1])

image_padded[1:-1, 1:-1] = img

for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            output[y, x] = (kernel2 * image_padded[y:y + largo, x:x + largo]).sum()


print(image_padded)
print(output)

plt.imshow(kernel, interpolation='nearest',cmap='gray')
plt.show()

plt.imshow(kernel2, interpolation='nearest',cmap='gray')
plt.show()
