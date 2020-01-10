# Importaciones
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

# Variables globales

#Kernel experimental 1
kernel_A = np.array([[256,256,256],[256,256,256],[256,256,256]])
#Kernel experimental 2
kernel_B = np.array([[250,250,250],[250,250,250],[250,250,250]])
#Kernel experimental 3
kernel_C = np.array([[0,0,0],[0,0,0],[0,0,0]])
#Kernel filtro de suavizado gaussiano
kernel_D = (1/256)*np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4.0],[6, 24, 36, 24 , 6],[4, 16, 24, 16, 4],[1, 4, 6, 4 ,1]])
#Kernel detector de bordes
kernel_E = np.array([[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]])


# Funcion: Genera una convolucion 2d entre una imagen y un kernel para aplicar un filtro que depende del kernel.
# Entradas: Arreglo de pixeles de una imagen, Arreglo de pixeles de un kernel
# Salida: Arreglo de pixeles de la imagen filtrada (a la que se le aplicó un kernel)
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


# Funcion: Generar el gráfico de una imagen en tonos grises con un título y los pixeles de esta en los ejes coordenados
# Entradas: Arreglo de pixeles de una imagen, titulo del gráfico, nombre eje x, nombre eje y
# Salida: Gráfico con imagen en tonos grises
def graph_convolution(img, title, ylabel, xlabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()

# Funcion: Mostrar una imagen en tonos grises con un título,
#          específicamente utilizado para mostrar la transformada de fourier de una imagen
# Entradas: Arreglo de pixeles de una imagen, titulo de la imagen
# Salida: Transformada de Fourier de una Imagen en tonos grises
def graph_fourier(img, title):
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap='gray')
    plt.show()

# Funcion: Calcular y centrar la transformada de fourier de una imagen.
# Entradas: Arreglo de pixeles de una imagen
# Salida: Arreglo de Transformada de fourier de una imagen
def fourier_image(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitudFFT = np.log(np.abs(fshift))
    return magnitudFFT


if __name__ == '__main__':
    # Se carga la imagen en tonos grises
    img = io.imread('leena512.bmp', as_gray=True)

    # Se muestra la imagen original
    graph_convolution(img,'Imagen original', 'Pixel', 'Pixel')

    # Se muestra el comportamiento de la imagen al aplicar el filtro que se obtiene con el kernel experimental 1
    graph_convolution(convolution(img, kernel_A),"Filtro Experimental 1","Pixel","Pixel")

    # Se muestra el comportamiento de la imagen al aplicar el filtro que se obtiene con el kernel experimental 2
    graph_convolution(convolution(img, kernel_B), "Filtro Experimental 2", "Pixel", "Pixel")

    # Se muestra el comportamiento de la imagen al aplicar el filtro que se obtiene con el kernel experimental 3
    graph_convolution(convolution(img, kernel_C), "Filtro Experimental 3", "Pixel", "Pixel")

    # Se muestra el comportamiento de la imagen al aplicar el filtro suavizado gaussiano
    graph_convolution(convolution(img, kernel_D), "Filtro Suavizado Gaussiano", "Pixel", "Pixel")

    # Se muestra la transformada de fourier del filtro suavizado gaussiano
    graph_fourier(fourier_image(convolution(img,kernel_D)),'Fourier Filtro Suavizado Gaussiano')

    # Se muestra el comportamiento de la imagen al aplicar el filtro detector de bordes
    graph_convolution(convolution(img, kernel_E), "Filtro Detector de bordes", "Pixel", "Pixel")

    # Se muestra la transformada de fourier del filtro detector de bordes
    graph_fourier(fourier_image(convolution(img,kernel_E)),'Fourier Filtro Detector de bordes')

