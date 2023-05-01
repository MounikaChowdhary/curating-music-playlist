# curating-music-playlist
curating a music playlist by detecting user emotions

ABSTRACT

The art of music has been recognized for its relevance to human feeling throughout history it has a singular ability to boost ones mood of late people have hectic lifestyles therefore stress has increased and people are more inclined to think about music whereas arts are their daily duty which helps them relax once a nerve-racking day as a result songs have become a necessary part of everyday fashion if a suggestion is given to the user which he likes it will increase his listening experience this project issues curating music playlists in context with user feelings it consists of four main phases face detection feature extraction mood detection and music recommendation emotion recognition applications need to localize and extract face regions from the rest of the pictures here we are generating a playlist from the users emotions to add a touch of the current situation of the users mood and personal choices of songs for a more personalized experience we use an artificial replication of human neural behavior for emotion detection and later classify the songs for song classification we have divided the songs into seven categories happy sad surprise neutral disgust fearful and angry the users video is captured in real-time and emotion is detected then playlist is suggested on the right side of the video frame on clicking the play button beside the song user is redirected to Spotify where the user can listen to the songs.


INTRODUCTION 
Every day every second, many people would listen to music at a time in this world it has become an important way of relieving stress in this modern period because of this trend, many music systems are being constructed after the development of AI, and ML music recommendation systems are growing these days, and with different methodologies, a new method has been proposed to understand and suggest music. 
It has many differences, and experiments from different fields predict that users' experience will increase if the songs are recommended according to their liking. In this project, we have another way of making song recommendations using facial expressions as they describe a person's mood. 
It is more effective than the manual work of searching and creating a specific playlist every time the facial expression aptly describes the person's mood a camera or webcam takes a picture of a user's face, and input is taken from that webcam input the picture determines the mood of a person, and it is used for validating a face from multi-media photos hence the demand for face identification has been essential worldwide there has been an increase in the applications for discovering and identifying faces it is used for many security essentials like public safety and other circumstances this design presents the two feature recognition ways haar cascade and local binary pattern.
 	In recent times, the music recommendation system that we go through daily is amazon and eBay, which offers suggestions based on user liking and past experiences. Companies like spotify and pandora employment machine literacy and deep literacy ways to give recommendations for substantiated music recommendation there are two avenues one is the content-grounded filtering approach and the cooperative filtering approach by assaying the content of music users have liked in the history content-grounded filtering makes recommendations for music that users are likely to enjoy through collaborative filtering a peer group of analogous tastes in music recommends music to each other the debit of content- grounded filtering is that the recommendation will be grounded on the users current interests as for the cooperative filtering approach it's the fashionability bias problem that constantly rated particulars get important exposure while lower popular ones are underrepresented in the recommendations hence a mongrel approach is enforced.

PROBLEM STATEMENT
	People find themselves relaxed when they listen to music, it helps them to deal with their day-to-day activities, and hectic lifestyle in a smooth manner.
	For e.g.: Engineering students prefer listening to music while writing assignments, working on projects and studying because it helps them reduce stress and be more productive.  

OBJECTIVES
•	In the view of the problem statement, we feel that providing a music playlist to people by detecting their mood will help reduce their work of manually searching for songs every time.
1.	To detect human emotions by capturing image through webcam in real time.
2.	After detecting the emotion, a song according to the user’s emotion will be recommended thus reducing the work of user to select songs every time.

 
PROPOSED SYSTEM
•	To achieve the main objectives of our problem, we are making use of real time facial emotion recognition.
•	Creation of an emotion recognition model to recognize the emotion from the user.
•	Once the emotion is identified, a music recommendation system is to be created to play the songs according to the emotion.

SYSTEM ARCHITECTURE
![image](https://user-images.githubusercontent.com/67090187/235411707-84ec47be-6a3b-47f5-957e-a65a1497a523.png)
First, the user turns on the webcam this is done to capture the users' emotions in real-time it is done with the help of the OpenCV library, which is a comprehensive open-source platform that enables users by providing various commands and processes for image and video operations with which we can capture a video from a laptop webcam in real-time with the help of a function VideoCapture(). This function helps capture user videos in real-time and from the system. In this case, we are using it in real-time since our project is about curating a playlist by recognizing emotion in real-time.
self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)

 In this the video capture object takes two parameters first, src is the input source to capture the video from which is the camera device index, and the second is cv2.CAP_DSHOW specifies the API to use for capturing video frames then the images from the video are converted to frames.

(self.grabbed,self.frame) = self.stream.read()

The read() method is called on the Video Capture object stored in the class variable self.stream. In addition to a boolean value indicating the frame being read successfully, the second is a numpy array of the image frame extracted from the video.


FEATURE EXTRACTION 
Haar Cascade Classifier:
Haar Cascade is an algorithm trained with a set of true and false scenarios and hence to train the models to help recognize features and thus extract features from it. It learns from the data it is trained with and applies those rules to recognize faces.
Models used are in xml format, taken from GitHub repository: https://github.com/opencv/opencv/tree/master/data/haarcascades.

Haar cascade works similar to convolutional kernel in terms of working as it produces a single feature by subtracting the addition of white region pixels with the addition of black region pixels. The face detection is done using three features:
i.	Edge Features
ii.	Line Features
iii.	Four-rectangle Features

The edge features detect features like eyebrows which have sharp edges by subtracting the white region above the eyebrow with black region below it.
The line feature detects features like nose, where there is white region in between two dark regions.
Four-rectangle features detect slant line features.
The Haar features traverse through the image.


EMOTION RECOGNITION:
Emotion Recognition is performed using CNN (Convolutional Neural Network).
CNN – Convolutional Neutral Network
CNN takes an input image. It is a Deep learning algorithm. It can differentiate between images. It requires less preprocessing than any other classification algorithms.
It can reduce images and make them easier to process. It is also done with the main features for prediction.
The CNN can learn from the training data, recognize the prominent features of each emotion, and classify new images or videos accordingly.


The input is a series of frames from the real-time video captured, which are preprocessed to extract facial features (using the Haar Cascade classifier), such as eyes, mouth, and eyebrows.
These features are then used as input to CNN which applies a series of convolutional and pooling layers to extract and learn features from the images. The output of the last layer is a probability distribution over the set of predefined emotions.




Convolution operation:  
Using convolutional operation, it is helpful in extracting features of high level which include edges, from the image input of user. 
Convolutional operation is carried out in Kernel.
A Kernel filters the features that are not important and it only focusses only on specific information. it moves along an image with stride (which defines the block scanned by the filter). This process is repeated until the entire image is traversed.     
Pooling operation:  
Size of the convoluted feature map is reduced using pooling. This will cause decrease in the computations that are required to process and extract features. 
Dropout: 
Dropout is done to avoid overfitting because if the accuracy of training is more that of the testing it will result in overfitting. In dropout we ignore neurons at the time of training, and they are not taken into account during a forward or backward pass
The feature vector from the feature extraction is fed to the CNN model, which is further processed by the application of filters and then the emotion is categorized into seven categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

Activation Layer: 
The activation layer applies a non-linear function to the output of the previous layer to introduce non-linearity into the model. The most common activation functions are ReLU, sigmoid, and tanh.

Batch Normalization Layer: 
The batch normalization layer is used to improve the stability and speed of training by normalizing the input to each layer to have zero mean and unit variance.

Flatten Layer: 
The flatten layer is used to convert the multidimensional output of the previous layer into a one-dimensional vector, which can be fed into a fully connected layer.

Fully Connected Layer: 
Fully connected layer also known as a dense layer, is a type of artificial neural network layer that connects every neuron in the previous layer to every neuron in the current layer.


MUSIC RECOMMENDATION:
A music recommender system is a system that learns from the user’s past listening history and recommends songs which they would probably like to hear in the future.
Recommendation systems are everywhere from e-commerce giants like Amazon, eBay, etc. Machine Learning and Deep Learning techniques are used by companies like Spotify, Pandora to provide personalized recommendations based on user taste and history.
TYPES OF RECOMMENDER SYSTEMS:
PERSONALIZED RECOMMENDER SYSTEMS: 
They are systems that adapt to the individual’s interests, desires, and preferences of every user.
1.	Content-based Recommender Systems:  CBRS recommends items based on their features and the similarity between elements of other items. 
2.	Collaborative filtering System: Every user and item is described by a feature vector or embedding for both users and items on its own. It embeds both users and items on same embedding space.
3.	Hybrid Recommender Systems: These systems have both content and collaborative techniques to extract maximum accuracy and to overcome drawbacks of both types.
NON-PERSONALIZED RECOMMENDER SYSTEMS: 
These systems do not take into consideration the preferences of users; they are made to be identical for every client.
SONG CLASSIFICATION:
Songs can be suggested for each emotion based on a combination of factors. Here, are some examples of these factors that contribute to the emotional resonance of a song:
•	Lyrics: The words of a song can evoke a range of emotions depending on their subject matter and the way they are delivered. For example, a song with lyrics about heartbreak and loss might be a good choice for someone feeling sad, while a song with lyrics about empowerment and self-confidence might be a good choice for someone feeling motivated.
•	Melody: The melody of a song can also contributes to its emotional impact. For example, a song with a slow, mournful melody might be a good choice for someone feeling melancholy, while a song with an upbeat, energetic melody might be a good choice for someone feeling happy or excited.
•	Tempo: The tempo, or speed, of a song can also influence its emotional impact. For example, a slow tempo might be associated with feelings of sadness or reflection, while a fast tempo might be associated with feelings of energy or excitement.
•	Rhythm: The rhythm of a song can also contributes to its emotional impact, as different rhythms can create different moods and feelings. For example, a song with a steady, driving beat might be a good choice for someone feeling determined or focused, while a song with a syncopated, unpredictable rhythm might be a good choice for someone feeling playful or adventurous.
•	Overall mood: The overall mood of a song can be a good indicator of its emotional resonance. For example, a song with a somber, introspective mood might be a good choice for someone feeling contemplative, while a song with a celebratory, festive mood might be a good choice for someone feeling joyful.
SPOTIFY API
To make the experience of the user more personalized we have integrated the music recommendation system with Spotify with the help of Spotify API. 
This not only provides a platform for the songs to be played but also provides an opportunity to take into consideration the previous likes of the user. Hence, helping in creating a personalized song listening experience to the user.
We purchased the developer pack of Spotify (Spotify premium) and used the credentials that is user Id and user secret key to integrate the web application to Spotify.



CONCLUSION
•	In this project, we have taken the user video with webcam and the emotion is detected by forming the boundary around the face. Our project even works for detecting emotions for multiple faces in single frame.
•	CNN model has been used for emotion detection and Haar cascade classifier has been used for feature extraction. 
•	The songs have been suggested beside the video feed after emotion is recognized. 
•	The Songs can be played by clicking on the play button beside each song.
•	The songs are played with the help of Spotify API which redirects the user to Spotify app and plays the song. 

FUTURE WORK
•	Further, we can deploy the website to provide the users with an open-source platform to access the application. In addition, we can add authentication to the website to prevent cyber-attacks.
•	Also, the accuracy of detecting emotions like disgust and fear fluctuates as they are primarily classified as angry since the facial features are similar. We can consider additional parameters like heart rate or body temperature to avoid mistakes.


