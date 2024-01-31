# Object Detection with ChatGPT Integration
This project is an object detection program with camera integrated to help blind people on detecting object in front of them.
Using latest YoloV8 model and over than 30 classes available, we hope this project can help on user to detect 
and know where are the objects in front of them. 

## Features
- Object distance and direction from camera 
- Voice assisted command from user using Whisper-AI
- YoloV8 Deep Learning detection Model
- 7 digit Pin user authentication
- MongoDB User Database
- Integrated ChatGPT with voice command as an AI assistent 
  
## Setup
### Prequisites
- Python 3.8 and up

### Internal Libraries
- threading
- time
  
### External Libraries
- OpenCV(cv2)
- ultralytics YOLO
- pyttsx3
- pymongo
- keras
- gc

## How to Run
- Install all the prequisites library
- Run the `finished.py` program using cmd or directly click the program
- Login or signup main menu to choose, 0 for login, 1 for signup
- Go straight into detecting program after login succesfull
- Program will automatically detect any object and say it out loud with text-to-speech
- Any keywords with `what` or `search` will automatically connect with chatgpt to answer the question
