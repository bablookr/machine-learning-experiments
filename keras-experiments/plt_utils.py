import matplotlib.pyplot as plt


def plot(x, y, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.plot(x, y, color='blue')
    plt.show()


def plot_scatter(x, y, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.scatter(x, y, color='blue')
    plt.show()


def plot_bar(x, y, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.bar(x, y, color='blue')
    plt.show()


def plot_image(img, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def plot_n_images(n_images, images, titles):
    for i in range(n_images):
        plt.figure(i+1, figsize=(10, 10))
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')

    plt.show()


def plot_1d_subplots(n_images, images, titles):
    fig, ax = plt.subplots(nrows=1, ncols=n_images, figsize=(10, 10))
    for i in range(n_images):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(titles[i])
        ax[i].imshow(images[i], cmap='gray')

    plt.show()


def plot_2d_subplots(rows, columns, images, titles):
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 10))
    for i in range(rows):
        for j in range(columns):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_title(titles[columns * i + j])
            ax[i, j].imshow(images[columns * i + j], cmap='gray')

    plt.show()