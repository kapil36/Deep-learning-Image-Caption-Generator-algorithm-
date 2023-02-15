# Deep-learning-Image-Caption-Generator-algorithm-
Deep learning Image caption generator algorithm providing a textual description of an image with vocal context

Introduction

Multi-image classification is the task of classifying multiple images into one or more predefined classes, here in this project we took flickr-8k dataset which included 8,000 multiple random images that are each paired with five different captions which provide clear descriptions of the salient entities and events as shown in these images. After pre-processing the caption using regex we splitted the images into a train-test split where 80% went for training and 20% for testing. We used densenet201 which  is a convolutional neural network that is 201 layers deep. CNN is used to extract features from the images and LSTM has been used for caption generation. We used 10 epochs in our model and eventually plotted the graph, from the graph we can see the validation loss gradually declines on increase of the number of epochs. We also added a text to speech feature in our project using the python package Tyttsx3. Using natural languages to automatically describe the content of photographs is a basic and difficult challenge. It has a significant potential impact. It may, for example, be beneficial.
People who are vision challenged grasp the content of photographs on the internet better. It might also deliver more accurate and concise picture/video information in contexts like image sharing in social networks or video surveillance systems.

The objective of our project is also to aid visually impaired to understand the images. The practical implementation of our project can be that if a visually impaired person has to understand the images, our model can explain to him/her what the image is all about.
Some examples of our dataset.

CNN Framework

In Deep Learning, a convolutional neural network (CNN/ConvNet) belongs to a class of deep neural networks that are frequently used to examine visual imagery. It makes use of a unique mathematics method called convolution. When two functions are combined mathematically, a third function is created that expresses how the shape of one is changed by the other.
The ConvNet's job is to simplify the images without sacrificing any of the features that are essential for making accurate predictions.	

![image](https://user-images.githubusercontent.com/64062901/218978755-887cff9e-cc3b-4c98-a064-69231498b88c.png)

How does CNN work?
A grayscale image is the same as an RGB image but only has one plane, whereas an RGB image three planes

![image](https://user-images.githubusercontent.com/64062901/218978808-7009d326-6bd7-4a26-adef-8fe4d4655e46.png)
![image](https://user-images.githubusercontent.com/64062901/218978831-6dd5692a-2a48-4e0d-9218-7310c742f0d8.png)

A convolution is depicted in the image above. To obtain the convolved feature, we apply a filter or kernel (3X3 matrix) to the input image. The following layer receives this convolved feature.
Artificial neurons are arranged in multiple layers to form convolutional neural networks. Each layer of a ConvNet creates a number of activation functions that are passed on to the next layer when an image is input.
Typically, the first layer extracts fundamental features like edges that run horizontally or diagonally. The next layer receives this output and detects more intricate features like corners or multiple edges. The network can recognise even more complex features, like objects, faces, etc., as we delve deeper into it.
The classification layer generates a set of confidence scores (values between 0 and 1) that indicate how likely it is for the image to be a member of a "class," based on the activation map of the final convolution layer. For instance, if you have a ConvNet that detects cats, dogs, and horses, the output of the final layer is the possibility that the input image contains any of those animals.

What’s a pooling layer?
The Pooling layer, like the Convolutional Layer, is in charge of shrinking the Convolved Feature's spatial size. By reducing the dimensions, this will lower the amount of computational power needed to process the data.
There are two types of pooling- average pooling and max pooling.

![image](https://user-images.githubusercontent.com/64062901/218978960-e7a7889f-0ba6-4f93-b2b6-9da96a394349.png)

In Max Pooling, we extract a pixel's maximum value from a selected area of the image. Additionally, Max Pooling functions as a noise suppressant. It also performs de-noising and dimensionality reduction in addition to completely discarding the noisy activations.
Comparatively, Average Pooling returns the average of all the values from the relevant area of the image covered by the Kernel. Dimensionality reduction is all that Average Pooling does to reduce noise. Therefore, we can conclude that Max Pooling outperforms Average Pooling significantly

RNN Framework:
The neural network style known as RNNs can be used to model sequence data. The behaviour of RNNs, which are built from feedforward networks, is comparable to that of human brains. Simply put, recurrent neural networks are better than other algorithms at anticipating sequential data.
In conventional neural networks, all of the inputs and outputs are independent of one another. However, there are times when prior words are required, such as when predicting the next word of a phrase, and it is therefore necessary to remember the prior words. RNN was developed as a result, and it used a Hidden Layer to solve the issue. The Hidden state, which retains specific information about a sequence, is the most crucial part of an RNN.

![image](https://user-images.githubusercontent.com/64062901/218979045-4a14be5d-f85c-4cc1-a68c-2b75d661e830.png)

With fewer parameters and the ability to generalize to sequences of various lengths, recurrent neural networks use the same weights for each element of the sequence. Due to their design, RNNs can be applied to structured data that is not sequential, such as geographic or graphical data.
![image](https://user-images.githubusercontent.com/64062901/218979126-3b2d5bc8-dbd9-401a-982c-71cf2e7ad1c4.png)

Some examples of RNN architectures:
•	One To One: There is only one pair here. A one-to-one architecture is used in traditional neural networks.
•	One To Many: A single input in a one-to-many network might result in numerous outputs. One too many networks are used in the production of music, for example.
•	Many To One:  In this scenario, a single output is produced by combining many inputs from distinct time steps. Sentiment analysis and emotion identification use such networks, in which the class label is determined by a sequence of words.
•	Many To Many: For many to many, there are numerous options. Two inputs yield three outputs. Machine translation systems, such as English to French or vice versa translation systems, use many to many networks.

How does Recurrent Neural Networks work?
![image](https://user-images.githubusercontent.com/64062901/218979254-d3ed047d-bf24-4bc8-8352-3b38e4afbbf7.png)

Recurrent neural networks have a loop where the information travels through before reaching the middle hidden layer.
Before sending the input to the middle layer of the neural network, the input layer x receives and processes it.
The middle layer h contains a number of hidden layers, each with a unique set of activation functions, weights, and biases. If the different hidden layer parameters are not affected by the hidden layer before it, or if the neural network has no memory, then you can use a recurrent neural network.
The Recurrent Neural Network will standardise the various activation functions, weights, and biases to guarantee that each hidden layer has the same properties. It will only create one hidden layer and loop over it as many times as necessary rather than building many.

![image](https://user-images.githubusercontent.com/64062901/218979418-11b5d294-67a8-4fd3-a7b7-6bc0b76637ca.png)

readImage feature
This method returns the picture as a NumPy array and accepts a file path path to an image and an optional image size parameter img size (the default value is 224). The function performs the subsequent actions:
The picture from the file at the specified location is loaded into a PIL image object using the load img function from the TensorFlow keras.preprocessing.image module. The target size parameter is set to a tuple of the intended image size (img size by img size), and the color mode parameter is set to 'rgb' to indicate that the picture is a 3-channel RGB image.
The TensorFlow keras.preprocessing.image module's img to array function is used to transform the PIL image object into a NumPy array of pixel values.
By dividing the array by 255, the pixel values in the NumPy array are scaled between 0 and 1.
It then returns the scaled NumPy array.
display_images function:
This function displays the photos in the image column together with their matching captions in the caption column, in a grid of 5 rows and 3 columns. It takes a pandas dataframe called temp df with columns image and caption. The function performs the subsequent actions:
The dataframe's index is reset using the reset index function such that it now starts at 0 and increases by 1 for each row. This helps to guarantee that the index is in the right order while iterating through the dataframe's rows.
Using plt.figure, a figure of the given dimensions (20 by 20 inches) is produced.
The dataframe's first 15 rows are iterated over in a loop. Each time the loop iterates:

With 5 rows, 3 columns, and the current loop index as the subplot number, the subplot is produced using plt.subplot.
The distance between the subplots can be changed by using the plt.subplots adjust method.
In order to read the picture from the current row of the dataframe and save it in the image variable, the readImage function is called.
The image is shown using the imshow function.
The description for the image is shown using the title function, while extended captions are split into numerous lines using the wrap function.
The axis labels and ticks can be disabled using the axis function.
After the loop has finished, plt.show is used to display the figure.

![image](https://user-images.githubusercontent.com/64062901/218979579-992417f5-176a-42c3-a542-9a77811a61b7.png)

providing a sample of 15 rows from a pandas dataframe to the display images function. The dataframe's rows are chosen at random using the sample function.
The example dataframe's photos in the image column and their matching captions in the caption column will subsequently be shown in a grid with 5 rows and 3 columns by the display images function. Using the procedures outlined in the previous response, the photos are read and presented

![image](https://user-images.githubusercontent.com/64062901/218979646-eb6a1a99-cff5-4bd8-a764-a7955eded1c8.png)

Lowercasing: Using the lower technique, all characters in the text are first converted to lowercase. This is frequently done to condense the text's language and strengthen the model's resistance to case changes.
Non-alphabetic character removal: Using a regular expression defined using the replace approach, the second step eliminates any non-alphabetic characters (A-Z or a-z) from the text. Any character that is not in the alphabet is matched by the regular phrase "[A-Za-z]". Punctuation and other special characters that are not necessary for the activity are frequently removed using this step.
Extra whitespace is eliminated in the third stage by utilising the replace technique to replace many whitespace characters with a single space. By eliminating non-alphabetic letters or any excess whitespace that may have been added during the lowercasing process, this step is helpful.
Getting rid of single-character words: The split approach and list comprehension are used in the fourth phase to get rid of single-character words. The text is divided into a list of words using the split technique, and only words with more than one character are included in the list comprehension iteration. Filler words and terms that are unlikely to be helpful for the purpose are frequently eliminated using this stage.
Adding special tokens: In the fifth stage, the join method is used to add the special tokens "startseq" and "endseq" to the start and end of the caption, respectively. These unique tokens serve as the input and output for a language model and are frequently used to denote the beginning and end of a run of text.

![image](https://user-images.githubusercontent.com/64062901/218979725-95812e97-d10d-4f3d-9c6b-916970ec4350.png)
Tokenizer:Using the fit on texts function, a Tokenizer object is generated and placed on the list of captions. The captions are tokenized in this stage by turning each caption into a list of numbers, each of which stands for a token (a word or a character) in the text.
word_index and tokenizer:By subtracting 1 to account for the 0th index from the length of the word index parameter of the Tokenizer object, the vocabulary size—the number of distinct tokens in the text—is determined.
Split:The maximum length of a caption is determined by adding up the longest captions in the list, where "length" is the split method's calculation of the amount of words in the caption.
Image:Using the unique and tolist methods, the list of unique image names is retrieved from the data's image column.
The length of the list of picture names is used to determine how many photographs there are.
isin:Round(0.85*nimages), where nimages is the number of images, calculates the index at which to divide the list of image names into training and validation sets.
resnet_index:Using the isin technique and the proper indices, the picture names are chosen from the list of image names to generate the training and validation sets.
Using the isin approach and the picture names in the training and validation sets, the training and validation sets are retrieved from the original dataframe.
Using the reset index method and the inplace and drop parameters both set to True, the indices of the training and validation dataframes are updated.
texts_to_sequence:The Tokenizer object's texts to sequences function may tokenize a single caption and deliver the results as a list of numbers.

![image](https://user-images.githubusercontent.com/64062901/218979804-f17624d6-d859-4280-8388-f4eacb778068.png)

DenseNet201:The DenseNet201 function from the TensorFlow keras.applications package is used to build a DenseNet201 model. DenseNet201 is a convolutional neural network (CNN) architecture that may be used to extract features from pictures. It was trained using the ImageNet dataset.
Model and fe:With the input layer set to the input layer of the DenseNet201 model and the output layer set to the second-to-last layer of the DenseNet201 model, a new model is constructed using the Model function from the TensorFlow keras module. The DenseNet201 model is used to the new model, fe, to extract features from photos.
The picture is set to 224 pixels wide.
The retrieved picture features are saved in a dictionary named features.
The data's unique picture names are iterated over using a for loop. Name of each image:
load_img:The load img method from the TensorFlow keras.preprocessing.image module loads the picture, which is then scaled to the desired image size (224 by 224).
The TensorFlow keras.preprocessing.image module's img to array function is used to turn the picture into a NumPy array of pixel data.
By dividing the array by 255, the pixel values in the NumPy array are scaled between 0 and 1.
Using the expand dims function, a new axis of size 1 is added to the NumPy array.

![image](https://user-images.githubusercontent.com/64062901/218979868-c80d18a8-9206-49a7-b5b8-86435eb2f1ad.png)

A custom data generator with the name CustomDataGenerator class is intended to produce data for model training and assessment. The model can learn from huge datasets that would not fit in memory thanks to the data generator, which generates batches of data that are supplied to it.
The following parameters are defined for the __init__ method, which initialises the data generator:
df: a dataframe holding the information the data generator will utilise. There should be at least two columns in this dataframe: one for the captions and one for the picture names.
X col is the name of the df column that has the names of the images.
the name of the column in the df that has the captions, y col
the amount of data points to include in each batch, or batch size.
directory: The location of the pictures' storage directory.
The captions are tokenized using a tokenizer object.
The vocabulary size, or the number of distinct tokens in the text, is known as vocab size.
max length: The maximum caption length, or the amount of words that can be included in a caption.
features: a dictionary with the extracted image features in which the features are the values and the keys are the names of the images.
A boolean value for "shuffle" determines whether to reorder the data after each epoch (default is True).
If self.shuffle is True, the on epoch end function is designed to shuffle the dataframe at the conclusion of each epoch.
The definition of the __len__ method is to return the number of batches in Two input layers are defined for the model using the Input function:
input 1: a layer for the image's characteristics with the form (1920).
input 2: A layer for the word order with the shape (max length,).
The TensorFlow keras.layers module's Dense and Activation functions are used to construct a dense layer with 256 units and a ReLU activation function that processes the image's features. This layer's output is known as img features.

![image](https://user-images.githubusercontent.com/64062901/218979997-055e854e-eb62-49fe-8df1-ea95070c619b.png)
During the model training process, batches of data are created using the CustomDataGenerator objects for training and validation. These objects offer a practical method for providing structured data to the model, enabling it to learn from big datasets that would not fit in memory.
With the following parameters, the train generator and validation generator objects are initialised:
df: a dataframe holding the information the data generator will utilise. There should be at least two columns in this dataframe: one for the captions and one for the picture names.
X col is the name of the df column that has the names of the images.
the name of the column in the df that has the captions, y col
the amount of data points to include in each batch, or batch size.
directory: The location of the pictures' storage directory.
The captions are tokenized using a tokenizer object.
checkpoint = ModelCheckpoint(model_name, monitor="val_loss", mode="min", save_best_only = True, verbose=1)
When the performance on the validation set improves, this generates a ModelCheckpoint callback that will save your model to the file model name each time. The performance metric that will be used to judge whether the model's performance has improved is specified by the monitor parameter. It is val loss in this instance, which stands for the model's loss on the validation set. The performance measure should either be reduced or maximised, depending on the mode option. It is set to min in this instance, indicating that the loss of the model need to be reduced. The save best only argument indicates whether to save all models or just the top model as determined by the performance metric. It is set to True in this instance, which means that only the best model will be kept. Whether or not to print notifications about the model being saved is determined by the verbose argument.

earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

This generates an EarlyStopping callback that will halt model training after a predetermined number of patience-specified epochs in which the performance on the validation set has not improved. As long as the model's loss on the validation set has not decreased for 5 epochs, training will end in this situation. The performance metric that will be used to judge whether the model's performance has improved is specified by the monitor parameter. It is val loss in this instance, which stands for the model's loss on the validation set. The performance metric's minimal change must occur for the min delta parameter to be considered an improvement. It is set to zero in this instance, indicating that any modification to the model's loss will be seen favourably. Whether or not to print notifications about the model halting is determined by the verbose option. When training is complete, the restore best weights option determines whether to reset the model's weights to their optimal values based on the performance metric. It is set to True in this instance, indicating that when training is over, the model's weights will be returned to their ideal levels.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=0.00000001)

When the performance on the validation set has not improved after a predetermined number of epochs, provided in patience, this triggers a ReduceLROnPlateau callback that will cut the learning rate of the optimizer. In this instance, patience is set to 3, which means that if the model's loss on the validation set has not decreased after 3 epochs, the learning rate will be decreased. The performance metric that will be used to judge whether the model's performance has improved is specified by the monitor parameter. It is val loss in this instance, which stands for the model's validation loss.

![image](https://user-images.githubusercontent.com/64062901/218980124-0c5bf537-ba3e-4fd0-942b-c9722ba1ea34.png)

The fit() function of the caption model object is used in this code to train a model. Data from a train generator object and a validation generator object are used to train and validate the model, respectively. The model's training period is determined by the epochs parameter. It is set to 10 in this instance, indicating that 10 training epochs would be used to train the model. A list of callbacks to be used during training is specified by the callbacks argument. The checkpoint, earlystopping, and learning rate reduction callbacks are utilised in this situation. The history of the model's loss and metrics during training is recorded in a History object that is returned by the fit() function. The history variable is given the History object.
An instance of a Python generator that produces batches of training data for the model is the train generator object. An instance of a Python generator that produces batches of validation data for the model is the validation generator object. When a model needs to be trained on a huge dataset that won't fit in memory, these generators are frequently utilised. They make it possible to train the model on the data in batches as opposed to everything at once.

![image](https://user-images.githubusercontent.com/64062901/218980184-1a5ecd55-51a8-48ef-91bd-d944165982cb.png)

A record of the model's losses throughout training and validation is kept in the history object. The loss values for both the training set and the validation set are contained in the dictionary that makes up the history.history property. Both the training set and the validation set's loss values are plotted using the plt.plot() method. The plot's title is specified via the plt.title() method. The y-axis label is set using the plt.ylabel() method. The x-axis label is set using the plt.xlabel() method. The loss values for the training set and the validation set are represented by the labels "train" and "val," respectively, in the legend that is added to the plot using the plt.legend() function. The plot is shown via the plt.show() method.
This plot may be used to check for overfitting and to see how well the model is doing during training. The loss on the training set will be reducing while the loss on the validation set will be rising if the model is overfitting.

![image](https://user-images.githubusercontent.com/64062901/218980231-62c8cea8-7a29-4e3a-924b-4b4426597389.png)

The term in the tokenizer's vocabulary that corresponds to the integer is returned by this function, which accepts an integer and a tokenizer object as input.
The dictionary that links vocabulary terms to integer indices is the tokenizer.word index attribute. The for loop repeatedly iterates over the key-value pairs in the dictionary, determining if the value (represented by the integer index) matches the input integer. If so, the appropriate term is returned. The method returns None if the dictionary does not contain the number.
When a model has been trained on text data and the input data has been tokenized using the tokenizer, this function may be used to change the model's integer prediction back to the original word in the vocabulary.

![image](https://user-images.githubusercontent.com/64062901/218980300-e435e285-7de6-4e68-babb-7d16c16ed1e8.png)

creates a caption for each image in the samples DataFrame by iterating over its rows.
To load the picture from the file system, use the load img() method. The size to which the picture should be enlarged is specified by the target size argument. In this instance, the picture is scaled down to (224, 224). The picture is changed from a PIL object to a NumPy array using the img to array() method. To scale the pixel values to the range [0, 1], the picture is then divided by 255.
To create a caption for the image, the predict caption() method is used. Caption model, image filename, tokenizer, maximum caption length, and image characteristics are all inputs for the function. The created caption is subsequently added to the samples DataFrame's 'caption' column for the current row.
This code presupposes the existence of a method named predict caption() that receives as inputs the caption model, the image filename, the tokenizer, the maximum length of the caption, and the image characteristics and outputs a string representing the produced caption. Additionally, it is assumed that the samples DataFrame has the columns "image" and "caption," and that "image" contains the filenames of the photos.

![image](https://user-images.githubusercontent.com/64062901/218980357-1b527c09-70fd-4f03-be79-18d8ed018926.png)

![image](https://user-images.githubusercontent.com/64062901/218980394-fc00ad5d-8cd3-4497-a923-d3750787828b.png)

![image](https://user-images.githubusercontent.com/64062901/218980416-2caddff3-f829-499e-8564-d68e6f26455b.png)

![image](https://user-images.githubusercontent.com/64062901/218980452-e870286b-7c17-4e62-9482-a95e59e078ef.png)

Text-to-speech capabilities is offered by the pyttsx3 package, enabling you to translate text strings into spoken words. You don't need to instal any additional software because it uses the text-to-speech engine that comes standard with the system.
You must first import the pyttsx3 library using the import statement before you can use it. The text-to-speech engine may then be started using the pyttsx3.init() method, and a text string can be spoken using the engine.say() function. The application is blocked while the text is being spoken thanks to the engine.runAndWait() method. The text-to-speech engine is stopped with the engine.stop() method.

In conclusion, picture caption generators are a useful tool for persons who are blind or visually impaired since they enable them to comprehend an image's meaning verbally. When given an image as input, these deep learning algorithms provide a textual description of the image that gives the image's content a vocal context.
The most popular method for creating an image caption generator is a blend of CNNs and LSTM networks. There are other methods as well. A sizable collection of photos and the written descriptions that go with them must be used to train the model. By feeding a picture into the model after it has been trained, the model may be used to provide a textual description of any image. For those who are blind, image caption generators can be used to describe a picture verbally on a website or in an application, or they can use computer vision to describe an image's content in real time. In general, picture caption makers are a useful tool for persons who are blind or visually impaired because they give them a method to communicate with and understand their surroundings more effectively. 
In general, picture caption generators are a crucial area of study and development in the field of assistive technology because they have the potential to dramatically increase the accessibility of visual information for people who are visually impaired. Image caption generators are expected to get better and more helpful for individuals who are sight impaired as deep learning algorithms develop.


