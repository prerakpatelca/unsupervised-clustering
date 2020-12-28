"""
For the data collected for the train_img.jpg for k vs inertia, there was no obvious “elbow” found. One can say that there's a little bit of an elbow found at k=6 and/or k=8. However, it's not very clear. I had multiple runs with different k values and there was no obvious value of "elbow" found. As there was no obvious elbow found, I used k=8 for task 2, I found it most appealing for test_img.

This program uses KMeans Clustering to cluster the natural image with different k values and then we uses the same the quantization to cluster the second image with the same the "RGB" value, which gives nice artistic look to it.

Prerak Patel, Student, Mohawk College, 2020
"""

import numpy as np
from skimage import data
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# loading training image using io library of skimage
image = io.imread("train_img.jpg")
# plotting the image using matplotlib
plt.figure()
# setting the title of the plot
plt.title("Original Image")
# populating the image to the plot
plt.imshow(image)
# turning off the axis for the plot
plt.axis('off')
# this method displays the plot
plt.show()
# reshaping the training image by converting it from 3d to 2d array to use it for the clustering
image_reshaped = image.reshape(549*644,3)

# loading testing image
test_image = io.imread("test_img.jpg")
# reshaping the testing image converting into 2d array from 3d
test_image_reshaped = test_image.reshape(918*1409,3)

# array of k values
clusters = [2,3,4,5,6,7,8,9,10,11,12]

# declaring the array to store inertia value for plotting
inertia = []

# looping through each array value in the clusters
for cluster in clusters:
    # declaring the KMeans with the number of clusters
    km = KMeans(n_clusters=cluster)
    # passing the training image data to KMeans fit function to cluster the data
    km = km.fit(image_reshaped)

    # condition to store check if the cluster value is 8 then we can pass the testing image data to predict function
    if(cluster == 8):
        # passing the testing data to the predict function of KMeans to quantize the testing image using the cluster values of training image
        test_km_pred = km.predict(test_image_reshaped)
        # declaring each pixel value using the cluster center RGB value
        test_quantized_pixels = km.cluster_centers_[test_km_pred]
        # shaping the image back to 3d from 2d array to plot
        test_quantized_image = test_quantized_pixels.reshape(918,1409,3)
        # converting the pixel value to int as the RGB value does not support float value
        test_quantized_image = test_quantized_image.astype(int)

    # declaring each pixel for the training image using the cluster center values
    quantized_pixels = km.cluster_centers_[km.labels_]
    # shaping an image back to 3d to plot
    quantized_image = quantized_pixels.reshape(549,644,3)
    # converting the pixel to int from float
    quantized_image = quantized_image.astype(int)

    # plotting each image with different k values
    plt.figure()
    plt.title("k = " + str(cluster))
    plt.imshow(quantized_image)
    plt.axis('off')
    plt.show()

    # printing out the k value and the Sum Squared Error(Inertia)
    print("k = " + str(cluster) + " SSE = " + str(km.inertia_))

    # appending the inertia value of each k value to the inertia array
    inertia.append(km.inertia_)

# plotting the figure for the different values of k for train_img vs inertia value
plt.figure()
plt.plot(clusters, inertia)
plt.title("train_img.jpg k vs inertia")
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()

# plotting the original test image
plt.figure()
plt.title("Original Test Image")
plt.imshow(test_image)
plt.axis('off')
plt.show()

# plotting the test image with quantization with k = 8
plt.figure()
plt.title("k = 8 quantized using train_img.jpg")
plt.imshow(test_quantized_image)
plt.axis('off')
plt.show()

