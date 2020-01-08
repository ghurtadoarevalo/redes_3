from skimage import io
from matplotlib import pyplot as plt
import numpy as np

kernel_A = np.array([[256,256,256],[256,256,256],[256,256,256]])
kernel_B = np.array([[250,250,250],[250,250,250],[250,250,250]])
kernel_C = np.array([[0,0,0],[0,0,0],[0,0,0]])
kernel_D = (1/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4.0],[6, 24, 36, 24 , 6],[4, 16, 24, 16, 4],[1, 4, 6, 4 ,1]])
kernel_E = np.array([[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]])
img = io.imread('leena512.bmp',as_gray=True)  # load the image as grayscale

def convolution(img,kernel):
    output = np.zeros_like(img)

    image_padded = np.zeros((img.shape[0] - kernel.shape[0] + 1, img.shape[1] - kernel.shape[1] + 1))

    # Se rota el kernel
    kernel_rotated = np.flipud(np.fliplr(kernel))

    # image_padded[1:-1, 1:-1] = img

    for x in range(kernel.shape[0] - 1, image_padded.shape[1]):  # Loop over every pixel of the image
        for y in range(kernel.shape[0] - 1, image_padded.shape[0]):
            output[y, x] = (kernel_rotated * img[y:y + len(kernel), x:x + len(kernel)]).sum()
    return output


def graph(img, title, ylabel, xlabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()


if __name__ == '__main__':
    graph(convolution(img, kernel_A),"Filtro A","Pixel","Pixel")
    graph(convolution(img, kernel_B), "Filtro B", "Pixel", "Pixel")
    graph(convolution(img, kernel_C), "Filtro C", "Pixel", "Pixel")
    graph(convolution(img, kernel_D), "Filtro D", "Pixel", "Pixel")
    graph(convolution(img, kernel_D), "Filtro E", "Pixel", "Pixel")

