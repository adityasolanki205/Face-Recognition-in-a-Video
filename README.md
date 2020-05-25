## Facial Recognition

[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)

This is a Facial Recognition application developed for **learning and implementation purpose only**. In this repository a model has been trained to detect and recognise faces of six individuals namely Aditya Solanki(Author), Ben Afflek, Madonna, Elton John, Jerry Seinfled, Mindy Kaling. The complete process is divided into 3 parts:

1. **Face Detection in a Photograph**.
2. **Implementation of FaceNet model on the extracted face**.
3. **Implementation of Linear Support Vector Machine to recognise the face**.


## Motivation
For the last one year, I have been part of a great learning curve wherein I have upskilled myself to move into a Machine Learning and Cloud Computing. This project was practice project for all the learnings I have had. This is first of the many more to come. 
 

## Tech/framework used

<b>Built with</b>
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)


## Code Example

```bash
    # clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/adityasolanki205/Face-Recognition.git
```

## Installation

Below are the steps to setup the enviroment and run the codes:

1. **Data Setup**: First the data setup has to be done. Download the [5 celebrity Dataset, Kaggle](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset). After the Download create one more sub folder in train and test folders for your own photos. Provide as diverse photos as you can find. 

2. **Face Detection**: Now we need to detect a face in the dataset. To do that we will use [Multi-Task Cascaded Convolutional Neural Network](https://arxiv.org/abs/1604.02878) (MTCNN) 

```bash
    # All the codes are written in Jupyter Notebooks

    # Install MTCNN
    !pip install mtcnn
    
    # To Preprocess the image install PIL 
    !pip install PIL
    
    # Preprocess the image into 'RGB' and convert it into numpy array
    image = np.asarray(image.convert('RGB'))
    
    # Use MTCNN object to detect faces using detect_faces() method
    faces = MTCNN.detect_faces(image)
```
This will provide the co-ordinates of pixels to identify face in the photo. Same process can be done to fetch more than one face from a photo with multiple people. 

3. **Face Embeddings**: After face extraction we will fetch the face embedding using [FaceNet](https://github.com/davidsandberg/facenet). The model can be downloaded from [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn).


## Tests
Describe and show how to run the tests with code examples.

## How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## Contribute

Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus.

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. 

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Yourname]()