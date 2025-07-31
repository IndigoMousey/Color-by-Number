import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


# KMeans clustering (MiniBatch)
def kmeansClustering(imgPath, nClusters=10):
    if not os.path.exists(imgPath):
        raise FileNotFoundError(f"Image at path '{imgPath}' not found.")

    # Read in our image
    im = cv2.imread(imgPath)[:, :, ::-1]

    # Turn the image into an array
    imArray = np.array(im)

    # Make the array 2-dimensional so it can be turned into a DataFrame
    rows, cols, _ = imArray.shape
    imArray2D = imArray.reshape((rows * cols, 3))
    df = pd.DataFrame(imArray2D, columns=["R", "G", "B"])

    # -----------------INFO-----------------
    # All 3 columns are uint8
    # print(df.dtypes)
    # Takes up 9579615 bytes to store this DataFrame
    # print(df.memory_usage(index=True, deep=True).sum())
    # We also can confirm our reshaping didn't mess up the data. The original image was 2159x1479x3 so we'd expect after reshaping to have a 3,193,161x3 DataFrame which is indeed the case
    # print(df.shape)

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=nClusters, n_init=40)
    kmeans.fit(df)

    # Note, the centers are most likely not integers, so we'll need to approximate them
    centers = np.round(kmeans.cluster_centers_).astype(np.uint8)
    labels = kmeans.labels_
    compressed_img_array = centers[labels]

    # Change the shape back to normal (Uses rows and column values from initial import)
    compressed_img = compressed_img_array.reshape((rows, cols, 3))

    return im, compressed_img

# For viewing the images next to each other
def side_by_side(original, compressed):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(compressed)
    ax[1].set_title("Compressed Image")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()


images = ["bird.jpg", "oriole.jpg"]
for i in range(len(images)):
    original, modified = kmeansClustering(images[i])
    side_by_side(original, modified)
