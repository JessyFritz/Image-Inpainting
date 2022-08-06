# Image-Inpainting
Recreating images from sparsely pixel data using ML.

The program trains a neural network to predict unknown parts of an image. Samples are RGB images for which a regular grid of pixels
of the images is known (all other pixels were set to zero).

**Note**: You have to create a folder called "images", which contains your own images. 
This folder will then be split into training and validation data. The program calculates random spacing (tuple specifying spacing between two successive grid points in x and y direction - max 6 pixels) and offset (tuple specifying offset between two successive grid points in x and y direction - max 8 pixels) as illustrated below.

![image](https://user-images.githubusercontent.com/39498906/183264105-c3123222-f8d7-41cb-b314-fd58ec212b80.png)

After some hours of training, you can expect predictions like these. On the left is the input to the network, where only a fraction of pixels is known, in the middle is the target image, and on the right is the network prediction.

![0015000_27](https://user-images.githubusercontent.com/39498906/183264181-41893326-93eb-4b9e-98bf-5e4ffad9deca.png)

