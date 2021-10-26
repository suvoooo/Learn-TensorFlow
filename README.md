# Tutorials for TensorFlow
### Influenced from the [Tensorflow Specialization](https://www.coursera.org/specializations/tensorflow-in-practice) Course. 
#### Many Extra Issues, Concepts and Visualizations are added.    
* _Conv_Basic_with_TensorFlow:_ Start with building blocks of Deep Neural Networks and Slowly Build and Understand Convolutional Neural Network. 
Below is an example of how Conv and Max Pool layer understand features, training images are from MNIST Fashion Data-Set.  
![Convunderstadning](https://github.com/suvoooo/Learn-TensorFlow/blob/master/Conv_Basic_with_TensorFlow/Understanding_conv2.png)
* _im_generator:_ Learn to Preprocess Image Using Keras, 'flow_from_directory' and augemntation on the fly. 
* _Multi_Class:_ Moving from binary classification to multi-class classification. 
* _Transfer_Learning:_ Using VGG16 weights to train a model. 2800 images of cats and dogs are used for training and 600 images for validation. Example of classifying images downloaded randomly from internet
![catsdogs](https://github.com/suvoooo/Learn-TensorFlow/blob/master/Transfer_Learning/Prediction_with_Vgg_CatsDogs.png)

* _NLP_Embedding:_ Moving on from computer vision tasks, here we discuss classifying IMDB movie reviews and visualizng how embedding can help cluster words (positive & negative). 
* _NLP_LSTM_Glove:_ Using pretrained Glove to classify tweets of positve sentiments from the negative one. [Dataset - 1.6M Tweets](https://www.kaggle.com/kazanova/sentiment140)
_Word Cloud of 50 Negative Words_--
![wd_cld_neg](https://github.com/suvoooo/Learn-TensorFlow/blob/master/NLP_LSTM_Glove/Wd_cld_neg_sent.png)
_Word Cloud of 50 Positive Words_ --
![wd_cld_pos](https://github.com/suvoooo/Learn-TensorFlow/blob/master/NLP_LSTM_Glove/Wd_cld_pos_sent.png)

* _Time_SeriesW#:_ Deal with time series data, starting from simple linear regression to RNN, LSTM, 1D Convolution were used to build deep networks. Example of predicting minimum temperature of Melbourne is shown below-- 
![min_temp](https://github.com/suvoooo/Learn-TensorFlow/blob/master/Time_SeriesW4/Min_Temp_Melbourne.png)

* _TFDataIntro.ipynb:_ Introducing [TF Dataset API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and how to use them effectively to build efficient and fast data analysis pipeline. More details in [TDS](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435). 

* _cassava-classification.ipynb:_ Cassava leaf disease classification competetion in [Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview). The notebook here is the same one used for submission in the competetion. 

* _detectpneumonia...ipynb_ Pneumonia detection using TensorFlow dataset API. The dataset is available in [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). Many important comcepts such as standardization of images, roc curve as performance metric were influenced by DeepLearning.ai [course](https://www.deeplearning.ai/program/ai-for-medicine-specialization/). 
