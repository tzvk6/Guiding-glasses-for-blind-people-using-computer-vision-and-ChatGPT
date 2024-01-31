# Guiding glasses for blind people using computer vision and ChatGPT
This project is an object detection program with camera integrated to help blind people on detecting object in front of them.
Using latest YoloV8 model and with 36 classes available, this project aims to empower visually impaired individuals. The project offers real-time object detection, distance measurements, and directional guidance, complemented by an integrated ChatGPT for answering questions.

## Features
- Object detection, distance and direction detection from camera 
- Voice assisted command from user using Whisper-AI
- YoloV8 Model trained on 36 object
- 7 digit Pin user authentication
- MongoDB User Database
- Integrated ChatGPT with voice command as an AI assistent 
  
## Setup
### Prequisites
- Python 3.8 and up
- installing ultralytics library
```
pip install ultralytics
```
- installing opencv-python library
```
pip install opencv-python
```
- installing pyttsx3 library

```
pip install pyttsx3
```
- installing pymongo library

```
pip install pymongo
```
- installing keras library

```
pip install keras
```
- installing gc library

```
pip install gc-python-utils
```



## How to Run
- Install all the prequisites library
- Run the `GP_Code.py` program using cmd or directly click the program
- Login or signup main menu to choose, 0 for login, 1 for signup
- Go straight into detecting program after login succesfull
- Program will automatically detect any object and say it out loud with text-to-speech
- Any keywords with `what` or `search` will automatically connect with chatgpt to answer the question
