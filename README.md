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
Use-Case Diagram
![image](https://user-images.githubusercontent.com/67090187/235412395-2851853a-edd8-41bc-8490-2ae465e496e5.png)

DATASET
•	Pierre-Luc-carrier and Aaron Courville prepared the FER-2013 image dataset as part of an ongoing project.
•	The dataset contains approximately 35,887 grayscale images. The dataset is further divided into the train and test data in an 80:20 ratio; 80 % data is put for training, and the rest 20% is put for testing.
•	The images in both train and test are labelled under seven categories: Angry, Happy, Sad, Disgusted, Fearful, Surprised and Neutral.
•	Training data – 28710 images:
	Angry – 3995
	Sad – 4830
	Happy – 7215
	Neutral – 4965 
	Disgust – 436 
	Fear – 4097 
	Surprise – 3171
•	Test data – 7195 images:
	Angry – 985
	Sad – 1247
	Happy – 1774
	Neutral – 1223 
	Disgust – 111
	Fear – 1024
	Surprise – 831
•	For playlist we used dataset from Kaggle for English and Hindi. In the song dataset there are 7 categories of emotions into which the songs have been classified they are: Angry, Happy, Sad, Neutral, Disgust, Fear and Surprise.
•	There are 492 English songs and 70 Hindi songs.

EMOTION DETECTION 
    Training and validating the data 

![image](https://user-images.githubusercontent.com/67090187/235412477-c9106fb3-5201-4a21-bddc-b96a00c6ccc1.png)
Fig. 6.1: Code for Training and Validating Data
   Line 54: load the training data
     Line 55: load the test data 

![image](https://user-images.githubusercontent.com/67090187/235412537-3db8ff94-a98a-47b0-b28a-68e5d6a0aabb.png)
Line 27: creating a model using the Sequential model is used for a plain stack of layers. Each layer has exactly one input and one output.
Line 28: we are using conv2D of Keras. It is a 2D convolutional layer. 
Here the activation function used is ‘relu’. We use ReLU in the hidden layer because it includes non-linearity to the input thus leading to avoidance of vanishing gradient problem.
Line 30: adding the Maxpooling2D, max pooling selects maximum element, the output will be a feature map which consists of important features
Pool_size = (2,2) pool size is the size of an input window
Line 31: Add the Dropout layer to the model. In this technique, during training, randomly selected neurons are ignored. Their contribution to downstream neurons for the activations is temporally removed on the pass.
While the neuron networks learn, the weights settle within the network. Neuron weights are tuned for specific features.
While the neuron networks learn, the weights settle within the network. Neuron weights are tuned for specific features.
If neurons are dropped randomly out of the network, the remaining neurons will step in and carry out the predictions. 
This will make the network work for all weights rather than specific weights. Hence, it is less likely to overfit the training data.
Line 37: adding flatten () function to the model 
Flatten function is used to flatten multi-dimensional input into a single dimension, so the input layer can be used to model neural networks. Pass the input to the neuron, and batch size is unaffected.
For example, when flatten is applied to a layer of input shape (batch_size,2,2), it results in the output shape of (batch_size,4). Flatten layer is placed at the end after all the convolutional layers because it converts 2D convolutional layers output into a linear vector. The linear vector is given as input to model to classify the image.

Line 40: model.add(Dense(7, activation=’SoftMax’ )dense layer is added. The dense layer is the regular deeply connected neural network layer. The dense layer offers learns features from all combinational features of the previous layer.

The SoftMax function is an activation function that predicts a multinomial probability distribution; hence it is used for multi-class classification problems where class membership is required on more than two class labels.

![image](https://user-images.githubusercontent.com/67090187/235412591-07ed47cf-b96d-4e4a-942c-36820d4b660d.png)
Line 87: gives the VideoCapture method followed by 2 parameters src which indicates the file path or source and cv2.CAP_DSHOW indicates the API used for the purpose. 

Line 88: The stream is read and output has 2 return values that is a Boolean value stating if the frame is captured or not (True or False) and other is the NumPy array of the features (feature map). The Boolean value is stored in self.grabbed and NumPy array is stored in self.frame which is used for further processing.

Line 103: Unless the thread is stopped the next frame is read from the stream and assigns the return values to self.grabbed and self.frame.
![image](https://user-images.githubusercontent.com/67090187/235412621-ac945848-c3ef-4363-bfe1-cd7953120ab6.png)
![image](https://user-images.githubusercontent.com/67090187/235412636-4cfe7b5e-99c8-4d2d-98e4-eb8c75698c2b.png)
Line 120: The WebcamVideoStream class is called with src=0 indicating the source of video stream to default camera and start method initializing the video stream and starts capturing frames from the camera
Line 121: The frame read by the cap1 is stored in the image variable.

Line 126: Image is converted to grayscale because grayscale images have only one color leading to less memory requirement, also less noise which important while performing tasks like emotion recognition and most importantly by reducing the amount of data that needs to be processed, operations on grayscale images can be faster than on those RGB images.

Line 128: Shows the English playlist on the left side of the video feed. 

Line 131: Detects the face and surrounds the face with the rectangle and displays the emotion above the rectangle.

Line 143: Checks for the time the frame has been active on if it has crossed the 20 seconds time lapse then it would start the detection of emotion again.
![image](https://user-images.githubusercontent.com/67090187/235412670-a774d068-00b4-4ef6-8d5c-ec1bc867a160.png)
Line 166 – 172: music_rec() method which is for English songs, upon being called it reads the music_dict on basis of emotion detected and then on the basis of index of the text shown it displays the dataframe of the playlist.

Line 173-179: music_rec_hindi() method which is for Hindi songs, upon being called it reads the music_dict on basis of emotion detected and then on the basis of index of the text shown it displays the dataframe of the playlist.
![image](https://user-images.githubusercontent.com/67090187/235412716-03471747-cdfa-4e20-866c-e93f4349b56f.png)
Line 13: App = Flask(__name__) creates the Flask instance. __name__ is the name of the current pytohn moudule. The apps needs to know where it’s located to set up some paths and __name__ is a convenient way to tell it that.
![image](https://user-images.githubusercontent.com/67090187/235412734-321dacf0-4121-47f3-b399-4fb30c2830a2.png)

Line 31: App routing is used to map the specific URL with the associated function that is intended to perform some task. The route() decorator in Flask is used to bind URL to a function. GET  is the most common method. A GET message is send and the server returns the data.To handle GET request we add thay in the decorator app.route() method.

Line 33: In Flask, we can use the request.args attribute of the request object to access the URL parameters. There parameters are appended to the end of the URL in the form of key=value, sperated by ampersands.
![image](https://user-images.githubusercontent.com/67090187/235412774-5cfebdc8-105b-4028-ad46-b95dec9ce795.png)
Line 52: The function call is render_template, which is a method provided by Flask for rendering HTML templates. It takes two arguments: the name of the HTML file to render (index.html), and a set of variables that will be passed to the template engine to render dynamic content in the HTML file. The first variable being passed is headings, which presumably contains a list of column headings or titles for a tabular data structure. The second variable being passed is df3, which presumably contains some sort of tabular data (like a pandas DataFrame or a list of lists) that will be displayed in the HTML file.

Line 54: The to_json() function takes several arguments, but in this case, it is only using the orient argument, which specifies the format of the JSON string. The orient argument is set to 'records', which means that the resulting JSON string will be a list of records, where each record corresponds to a row in the DataFrame.
![image](https://user-images.githubusercontent.com/67090187/235412795-f376c5b9-22aa-408c-ac98-ea1943343b3d.png)
Line 66: @app.route('/video_feed') is a decorator in Python Flask framework that defines the URL endpoint for serving a video feed or stream.

Line 67: The video_feed() function is the view function that will be executed when a user accesses the URL endpoint '/video_feed'. This function returns a Response object that will generate the video feed.

Line 69: gen(VideoCamera()) is a generator function that captures frames from the VideoCamera object and generates a continuous stream of video frames.



Line 70: mimetype='multipart/x-mixed-replace; boundary=frame' is a parameter that sets the MIME type of the response. In this case, it is set to 'multipart/x-mixed-replace; boundary=frame', which is a standard format for serving real-time video streams.
•	Overall, the video_feed() function creates a Response object that generates a continuous video stream from the VideoCamera object, and the @app.route('/video_feed') decorator sets the URL endpoint for accessing this video stream.

Line 71: This code snippet appears to be using a Flask web framework to define a route at /get_access_token. When a GET request is sent to this route, it retrieves an access token using an object called sp_oauth, likely related to the Spotify API OAuth authentication flow.

Line 73: The get_access_token method is called on the sp_oauth object, which appears to retrieve the access token from some authorization server. The access token is then returned as a JSON response with a key of access_token and its corresponding value.
![image](https://user-images.githubusercontent.com/67090187/235412827-9e025964-d5d1-4d3f-8915-1a1030840db3.png)
Line 102: This code snippet appears to define a route at /spotify using the Flask web framework, with the HTTP method of GET

Line 105: When a GET request is sent to this route, it retrieves two query parameters, artist and song, using the request.args.get() method. The values of these query parameters are assigned to the artist and song variables, respectively.

Line 108: The SpotifyOAuth class provides a way to authenticate and authorize access to the Spotify Web API using OAuth2 authentication.

Line 109: The client_id parameter is a unique identifier assigned to the application or service that is making the API requests, which is required for authentication purposes.

Line 110: The client_secret parameter is a secret key used to authenticate the client application and must be kept secure.

Line 111: The redirect_uri parameter is the URI to which the Spotify API server should redirect the user after the user grants/denies permission to the client application

Line 112: scope parameter is set to user-read-playback-state and user-modify-playback-state, which indicates that the client application is requesting permission to read and modify the current playback state of the user's Spotify account.

Line 116: This code snippet appears to call the get_access_token() method on the sp_oauth object to retrieve an access token for the Spotify Web API. The get_access_token() method likely uses the OAuth2 authentication flow to obtain an access token, which can be used to make authenticated requests to the Spotify API on behalf of the user.

Line 119: This code snippet appears to create an instance of the Spotify class from the Spotipy Python package, which is a Python client library for the Spotify Web API. The auth parameter is set to the access_token variable, which likely contains an OAuth2 access token retrieved from the Spotify Web API using an authentication flow like the one provided by the SpotifyOAuth class.

Line 122: This code snippet appears to use the ‘sp’ object, which is an instance of the ‘Spotify’ class from the Spotipy Python package, to make a search request to the Spotify Web API.
•	The search() method is called on the sp object with the following parameters:
•	q: This is the query parameter and it is set to a string formatted as artist:{artist} track:{song}. This specifies the search query and indicates that the API should search for a track with the given song name and the artist name.
•	type: This is the type of object to search for and it is set to 'track'. This parameter limits the search results to only tracks.
![image](https://user-images.githubusercontent.com/67090187/235412893-d2d78719-6a1f-470a-8c24-eb5870f2ef87.png)
Line 124: This code is a Python function that searches for a song on Spotify based on the artist and song title, and if a track is found, redirects the user to the Spotify web player to play that track.
•	The first if statement checks if the search returned any results. If the length of the list of tracks returned by the search is zero, the function returns the message "No tracks found for that artist and song"

Line 130: Otherwise, the function extracts the URI (Uniform Resource Identifier) of the first track in the search results list. The URI is a unique identifier for the track on Spotify.

Line 132: Finally, the function redirects the user to the Spotify web player using the URI, which is appended to the URL of the web player. The URI is extracted from the Spotify API response using Python's string manipulation capabilities, by splitting the URI string at the colon character ":" and selecting the last element of the resulting list using the index [-1].

![image](https://user-images.githubusercontent.com/67090187/235412935-fb895408-cda4-4765-a465-5115fa599131.png)
Line 136: This code defines a Flask route for generating a table in JSON format. The function gen_table() is called when the route /t is accessed.

Line 137: If the variable temp is True, the function calls the music_rec_hindi() function, which presumably generates a pandas DataFrame containing music recommendations in Hindi language, selects the first 15 rows, and returns the DataFrame in JSON format using the to_json() method with orient='records' argument.

Line 144: If temp is not True, the function returns another pandas DataFrame df1 in JSON format using the same to_json() method with orient='records' argument.

RESULTS
•	Emotion recognition is performed using CNN with Haar Cascade classifier resulting in an accuracy of 87.08%.
•	The web application shows video feed capturing emotions of the user on the right side and displays the Playlist recommended according to emotion detected on the left side. 
•	The emotion detection takes place with a time lapse of 20s that is it takes an emotion and suggests the playlist for 20 sec before it starts detecting emotion again.
•	Along with suggesting the music playlist it also has a play button beside the songs which redirects the user to Spotify and plays the song.



CONCLUSION
•	In this project, we have taken the user video with webcam and the emotion is detected by forming the boundary around the face. Our project even works for detecting emotions for multiple faces in single frame.
•	CNN model has been used for emotion detection and Haar cascade classifier has been used for feature extraction. 
•	The songs have been suggested beside the video feed after emotion is recognized. 
•	The Songs can be played by clicking on the play button beside each song.
•	The songs are played with the help of Spotify API which redirects the user to Spotify app and plays the song. 

FUTURE WORK
•	Further, we can deploy the website to provide the users with an open-source platform to access the application. In addition, we can add authentication to the website to prevent cyber-attacks.
•	Also, the accuracy of detecting emotions like disgust and fear fluctuates as they are primarily classified as angry since the facial features are similar. We can consider additional parameters like heart rate or body temperature to avoid mistakes.


