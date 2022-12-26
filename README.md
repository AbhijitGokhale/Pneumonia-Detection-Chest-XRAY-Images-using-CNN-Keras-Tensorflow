# Pneumonia-Detection-Chest-XRAY-Images-using-CNN-Keras-Tensorflow
**An end-to-end machine learning &amp; AI pipeline that uses X-ray images of the lungs to detect pneumonia in patients.**

## INTRODUCTION & PROJECT OBJECTIVE
Deep learning for the medical image classification is not only a topic of hot research but is a key technique of computer-aided diagnosis systems today.

According to WHO, every year over 150 million people are infected with pneumonia particularly kids below the age of 5 years. One in three deaths in India are caused due to pneumonia as reported by the World Health Organization (WHO). Chest X-rays are at the moment, the best available method for diagnosing pneumonia, and therefore play a crucial role in diagnosing and providing clinical care to the ones affected. 

However, detecting pneumonia in chest X-rays is a challenging task that relies on the availability of expert radiologists. Experts are either not available in remote areas or most people can’t afford it. Under such circumstances, automating the detection of diseases through AI becomes the need of the hour. This study will result into aiding healthcare practioners, physicians, doctors, hospitals to take quick actions if the chest Xray detects Pneumonia. (Based on recent studies it's been observed that pneumonia patients are more prone to have COVID symptoms and their ill-effects.)

We’ll build an end-to-end machine learning & AI pipeline that uses X-ray images of the lungs to detect pneumonia in patients.

## LIBRARIES
We have imported OpenCV for preprocessing and loading. For the image classification modeling part, we’ll be using Keras with Tensorflow as a backend.

![image](https://user-images.githubusercontent.com/84480824/209521483-d15d6405-20c8-42a0-a54f-5aab7da696f6.png)

## DATASET

* The dataset is taken from [Kaggle Competition](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)

* We will split the dataset into three sets - train, validation, and test. Let’s define the paths where our data is stored. There are three separate directories for train, validation, and test data. In each of these directories, there are two folders- one containing pneumonia x-ray images and the other containing normal x-ray images.

* Train data distribution
![image](https://user-images.githubusercontent.com/84480824/209521811-e3e38957-0608-4e1b-b747-05d1df3a5fe6.png)

As we can see from the above visualization that data for patients with pneomonia chest xrays are 3 times more than patients with normal chest xrays. This data imbalance could be a problematic to accurately train and validate the model.

![image](https://user-images.githubusercontent.com/84480824/209521985-7184284a-0892-4154-9751-af425dbc2285.png)

![image](https://user-images.githubusercontent.com/84480824/209522055-4a3e4f1e-1a76-4bda-8862-a00f55234336.png)

## Image Preprocessing Steps
Preprocessing is essential to transform images in a format that can be easily understood by the model and also to make the algorithm work more efficiently.

The different preprocessing steps that we’ll use here are:

* Since the images are of different lengths and widths, resize them to 224,224,3.
* Some images are in greyscale (1 channel), therefore convert them to 3 channel
* Images read using cv2 are in BGR format(by default), convert it to RGB.
* Normalize the image pixels by dividing them by 255 (an essential math trick for better performance).
* to_categorical is used to convert labels to one-hot encoded format.

## CNN MODEL

![image](https://user-images.githubusercontent.com/84480824/209522361-55d13904-8ead-45b8-b0fa-b5d334c40dbf.png)

The basic building block of any model working on image data is a Convolutional Neural Network. Convolutions were designed specifically for images.
There is a filter or weights matrix (n x n-dimensional) where n is usually smaller than the image size. A multiplication or dot product is taken of this matrix with the filter size patch of the input. The filter is applied systematically to each overlapping part or filter-sized patch of the input data, moving from left to right and then top to bottom. The result of this dot product between two matrices is a single value and through repetition of this process on different input patches, we get a matrix in the end.

There is also a bias value that is added after every dot product. The weight matrix and the bias value are the parameters of the neural network that are updated throughout training. Stride is the value by which the filter shifts on the image.

In a convolutional layer, there are multiple filters- this value is decided and fed by the developer when defining a layer. The use and significance of these convolutions might not be intuitive at first- it is hypothesized that they learn different things at different stages. The convolutions in the earlier layers learn to detect abstract things like edges, textures, etc. Towards the final layer, they learn to detect more specific objects based on classification categories.

A ReLu activation is applied after every convolution to transform the output values between the range 0 to 1. Max pooling is used to downsample the input representation. It helps the model to deal with overfitting by providing an abstract representation and also reduces the computational cost. The way Max Pooling works can be illustrated by the image below:

![image](https://user-images.githubusercontent.com/84480824/209522634-9404b0da-1e77-43a3-a690-dce44bf69a67.png)

We’ll use binary cross-entropy as our loss function because we have only 2 classes. Rmsprop will be our optimizer function.

## PREDICTION & MODEL EVALUATION
![image](https://user-images.githubusercontent.com/84480824/209522740-cda3cd2d-9bd6-474d-b054-a86f6f1c9ca3.png)

![image](https://user-images.githubusercontent.com/84480824/209522779-4f87c840-568a-4d38-9ad6-f15b38b7153f.png)

* Precision is the fraction of relevant instances among the retrieved instances. In our case, it is the number of people actually having pneumonia divided by all those predicted by the model as having pneumonia.
* Recall on the other hand refers to the relevant instances that were retrieved. Here, it is the fraction of people actually having pneumonia and are predicted positive by the model to the total number of people having pneumonia. It measures the potential of a test to recognize patients with the disease.
* F1 score is just the harmonic mean of precision and recall.

## RESULT & CONCLUSION

The y-axis of the chart is for true labels and the x-axis is for predicted ones.

* The number of people who are actually Normal and are predicted as Normal by our model is 74. These cases are called True Negatives.
* The number of people with Pneumonia but diagnosed as Normal are called False Negatives and there are just 2 patients for that.
* The number of people who were Normal but are diagnosed with Pneumonia by the model are called False Positives and these cases are 160.
* The number of people with Pneumonia who are also diagnosed with Pneumonia by the model are True Positives, these cases are 386.
* While training an ML algorithm to diagnose whether a patient has a disease or not, it is far more fatal to predict “Normal”

**For a person who actually has the ailment when compared to the other type of error i.e. predicting Pneumonia for Normal Patients. Thus, while training our aim should be to minimize False Negatives and we have successfully done that.**



