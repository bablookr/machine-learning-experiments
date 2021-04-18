from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


def load_image(image_path):
    img = load_img(image_path)
    plt.imshow(img, cmap='gray')
    plt.show()


load_image('data\Lenna.png')
