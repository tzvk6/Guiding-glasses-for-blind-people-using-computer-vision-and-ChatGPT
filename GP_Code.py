from whisper_mic import WhisperMic # require to install first
import pyttsx3 # require to install first
import time
import pymongo # require to install first
import openai  # require to install first
import cv2     # require to install first
import numpy as np  # require to install first
from ultralytics import YOLO    # require to install first
import threading
import keras    # require to install first
import keras.backend as K 
import gc # require to install first

tokenList = [] # Array containing pin from voice input
integer_number = [0,1,2,3,4,5,6,7,8,9] # All possible integer in a array
string_number = ["zero","one","two","three","four","five","six","seven","eight","nine"] # String integer in a arra

client = pymongo.MongoClient("mongodb+srv://moh:12345@mohammed.lvr6bpx.mongodb.net/?retryWrites=true&w=majority")

keywords = ["search","what"] # Keywords for Speech Recognition
model = YOLO('best.pt') # GP Yolov8 Model
# All of the classes names
class_names = ['Bathroom door', 'Bathroom entrance', 'Clothes hanger', 
           'Coffee machine', 'Food machine', 'Hand soap', 'PC desktop', 
           'Person', 'Snack machine', 'Tissue box', 'water machine', 
           'Workers- area', 'break chair', 'chair', 'class chair', 
           'class door', 'down ladder', 'elevator', 'elevator keypad', 
           'fire extinguisher', 'flower', 'hand wash', 'in and out door', 
           'ladder door', 'light buttons', 'mosque', 'paper trash can', 
           'person', 'sub-route', 'teacher podium', 'teacher table', 
           'trash can', 'up ladder', 'wall pillar', 'water faucet', 'white board']
# The width from the current class
class_width = [0.75, 0,7, 0.42, 0.25, 0.2, 0.1, 0.2,
               0.42, 0.85, 0.12, 0.3, 1.6, 1.6, 0.45, 0.38, 0.8,
               0.35, 1.2, 0.1, 0.1, 0.2, 0.7, 1.4, 0.8, 0.5,
               15, 0.35, 0.42, 0.5, 0.65, 1.4, 0.5, 0.6, 0.7, 1.4]
object_on_vision_counter = [] # Counter for the detected object on vision, each of the item have 0 in initial
object_on_vision = []   # Item detected on the camera
current_object_on_vision = []   # Current item detected on the current iteration
search_start = False      # Status for OpenAI
direction = ""            # Direction of the object
distance = 0              # Distance of the object

# Voice Notification with Text to speech, msg as the String that The machine will say out loud
def voice_notification(msg):
    engine = pyttsx3.init()
    newVoiceRate = 170
    engine.setProperty('rate',newVoiceRate)
    engine.say(msg)
    engine.runAndWait()

# Voice Notification for Object Distances
def voice_notification_object(obj_name, direction, distance):
    engine = pyttsx3.init()
    text = ""
    if obj_name is None:
        text = "Object is not found" 
    else:
        text = "{}, {}, {:.2f} meters away.".format(obj_name, direction, distance)
    newVoiceRate = 145
    engine.setProperty('rate',newVoiceRate)
    engine.say(text)
    engine.runAndWait()

# Voice Notification for ChatGPT answers
def voice_searchengine_notification(msg):
    global search_start
    search_start = True
    engine = pyttsx3.init()
    newVoiceRate = 145
    engine.setProperty('rate',newVoiceRate)
    engine.say(msg)
    engine.runAndWait()
    search_start = False

# Change all string item into an integer from an array or a string
def speech_to_text(string):
    for item in string:
        if item in string_number:
            string[string.index(item)] = (integer_number[string_number.index(item)])
    return string

# Remove a special character from a string with "character_to_remove" parameter
def remove_special_character(input_string, character_to_remove):
    return input_string.replace(character_to_remove, '')

# Function to take the question only from a speech recognition
def remove_unused_word(sentence):
    words = sentence.split()
    # Remove all "is, the, and a from the speech detected text"
    words = [item for item in words if (item != keywords[0] and item != keywords[1] 
            and item != "is" and item != "the" and item != "a")]
    counter = 0
    # Also remove "?" from the detected text
    for i in words:
        if "?" in i:
            words[counter] = remove_special_character(words[counter],"?")
        counter +=1

    words = ' '.join(words)
    return words

# Detect which value is a key from, from a dictionary function
def get_keys_by_value(dictionary, target_value):
    return [key for key, value in dictionary.items() if value == target_value]

# Take the integer from the PIN ID
def take_current_pin_number(input_string):
    numeric_part = ""
    for char in input_string:
        if char.isdigit():
            numeric_part += char
        # Convert the resulting string to an integer
    numeric_value = int(numeric_part)
    return numeric_value

# Function to encode the direction
def direction_x(x_dir,frame_width):
## Take the center of the detected object pixel, and the total width of the image, then divide it into 7 segments ##
    if x_dir < (frame_width/7 * 1 + frame_width/15):    # 1st segment, 9 o'clock
        direction = "9 o'clock"
    elif x_dir >= frame_width/7 * 1 + frame_width/15 and x_dir < (frame_width/7 * 2 + frame_width/15):  # 2nd segment, 10 o'clock
        direction = "10 o'clock"
    elif x_dir >= frame_width/7 * 2 + frame_width/15 and x_dir < (frame_width/7 * 3 + frame_width/15):  # 3rd segment, 11 o'clock
        direction = "11 o'clock"
    elif x_dir >= frame_width/7 * 3 + frame_width/15 and x_dir < (frame_width/7 * 4 + frame_width/15):  # 4th segment, 12 o'clock
        direction = "12 o'clock"
    elif x_dir >= frame_width/7 * 4 + frame_width/15 and x_dir < (frame_width/7 * 5 + frame_width/15):  # 5th segment, 1 o'clock
        direction = "1 o'clock"
    elif x_dir >= frame_width/7 * 5 + frame_width/15 and x_dir < (frame_width/7 * 6 + frame_width/15):  # 6th segment, 2 o'clock
        direction = "2 o'clock"
    elif x_dir >= frame_width/7 * 6 + frame_width/15 and x_dir < frame_width/7:  # 6th segment, 3 o'clock
        direction = "3 o'clock"
    else:
        direction = "None"
    return direction

# time.sleep alternatives using while loop
def delay_with_while_loop(seconds):
    start_time = time.time()
    elapsed_time = 0

    while elapsed_time < seconds:
        elapsed_time = time.time() - start_time

# Voice object in vision function, if an object is not in front of the camera, add a counter on it, if the counter exceeds 4, remove the object from the array
def voice_object_in_vision(current_object_on_vision,object_on_vision_counter,object_on_vision):
    global search_start
    if search_start != True:
        for cur_item in current_object_on_vision:   
                if cur_item not in object_on_vision:   # Add new object into the array if the item isn't in the array
                    object_on_vision.append(cur_item)
                    object_on_vision_counter.append(0)
                    voice_notification_object(class_names[cur_item],direction,distance)

        for cur_item in object_on_vision:
            index = object_on_vision.index(cur_item)

            if cur_item not in current_object_on_vision:        # Add a counter if an item missing in the camera
                    if (object_on_vision_counter[index] < 4):
                        object_on_vision_counter[index] += 1
                    else:                                       # Remove the item from the array if the counter exceeds 4 
                        object_on_vision.remove(cur_item)
                        object_on_vision_counter.pop(index)
            else:       # Reset the counter if the object is detected again on the camera
                object_on_vision_counter[index] = 0

# Voice Command Function
# 4th Function to Land, will detect speech and keep detecting for "search" and "what" keywords for the ChatGPT search
def voice_command():
    global search_start
    mic = WhisperMic(english=True,model="tiny",energy=1000)
    while True:
        if search_start == False:
            print("Please speak something...")   
            mic.energy = 1500
            text = mic.listen(timeout=1)
            text = text.lower()
            print(text)
            voice_object_in_vision(current_object_on_vision,object_on_vision_counter,object_on_vision)
            if "search" in text or "what" in text:      # If the keyword for the search in the detected text, search with chatgpt
                answer = search_engine(remove_unused_word(text))       
                voice_searchengine_notification(answer)
                search_start = False

# Video to image function, encoded every detected object
# 4th function to land, will keep looping and detect every object on the camera
def video_to_images(video_path):
    cap = cv2.VideoCapture(video_path)
    global current_object_on_vision
    global object_on_vision_counter
    global object_on_vision
    global distance
    global direction

    if not cap.isOpened():
        print(f"Error: Could not open the video file {video_path}")
        return

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        results = model.predict(frame, stream=False, verbose=False)
        _, frame_width, _ = frame.shape
        
        # Reset the current object on vision
        current_object_on_vision = []
        for r in results:   # Iterate every detected object
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]    # Take the x,y coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls)
                current_object_on_vision.append(cls)    # Add the detected object into the current_object array
                real_width = class_width[cls]

                camera_width = x2 - x1     
                camera_center = (x2-x1)/2 + x1
                direction = direction_x(camera_center, frame_width)     # Check the direction of the object from the camera
                distance = (real_width * frame_width) / camera_width    # Check the distance of the object from the camera
                mid_x = x1 + (x2-x1)//2                                 # Take the middle coordinates
                distance_str = "{:.2f} meters".format(distance)
                cv2.putText(frame, class_names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display the name of the object
                cv2.putText(frame, direction, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)   # Display the direction of the object
                cv2.putText(frame, distance_str, (mid_x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   # Display the distance of the object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)     # Put the rectangle on the detected object
        
            if not ret:
                break

        # Clear memory for the next iteration
        K.clear_session()
        gc.collect()
               
        # Display the frame 
        cv2.imshow('Video Stream', frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


# Function to integrate and search with ChatGPT
def search_engine(word):
    client = openai.OpenAI(api_key="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")
    content = "What is a " + str(word) + ", explain it with a simple sentence and a maximal of 2 sentences"
    assistant = client.beta.assistants.create(
        name="Explanatory",
        instructions="You are a an explanatory. You will help users to explain an object that are asked",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview",
    )

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=content,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Address this user as Mohammed.",
    )

    print("checking assistant status. ")
    print("in progress...", end="")

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == "completed":
            print("done!")
            messages = client.beta.threads.messages.list(thread_id=thread.id)

            print("messages: ")
            for message in messages:
                assert message.content[0].type == "text"
                if message.role == "assistant":
                    return message.content[0].text.value

            client.beta.assistants.delete(assistant.id)
            return None
        else:
            print(".", end="")

# Signup Function
# 2nd Function to Land, either keep looping until the signup is correct, or go back into the start function if signup is successfull
def signup():
    global db
    global collection
    global tokenList
    db = client['GP']      # Take the cluster from a mongodb Cloud
    collection = db["ob"]       # Take the database from a mongodb Cloud
    cursor = collection.find()      # Find all the database from the collection variable
    new_doc = {}
    # Iterate through the cursor to take all the database into a dictionary
    for document in cursor:
        new_doc = document      
    mic = WhisperMic(english=True,model="tiny",energy=1000)     # Start the mic, with the tiny version of the model and the energy to 1000 (the higher it is, the less sensitive the mic from a noise)
    text = "Please register a 7 digit number"
    voice_notification(text)
    print("Please register a 7 digit number...")   
    while True:     
        tokenList = []
        mic.energy = 1500       # Set it into 1500 to make sure noise is minimum
        text = mic.listen(timeout=1)           # Listen into the mic, give a timeout of 1 second
        text = text.lower()                    # Change all of the Speech to text string into lowercases 
        if ("timeout" not in text):
            # Remove all of the special characters from the detected text
            text = remove_special_character(text,".")
            text = remove_special_character(text,",")
            text = remove_special_character(text,"!")
            text = text.split() # Split the test into an array
            tokenList= speech_to_text(text)     # Change all string output into an integer
            print(tokenList)
            if (len(tokenList) > 7):        # Check if the token is higher than 7
                msg = "Input is higher than 7 digit, please try again"
                voice_notification(msg)
            elif (len(tokenList) < 7):      # Check if the token is less than 7
                msg = "Input is lower than 7 digit, please try again"
                voice_notification(msg)
            else:                           # If the token equals to 7
                loginToken = "".join(tokenList)     # Join all arrays into a string
                id = get_keys_by_value(new_doc, loginToken) # Check if the token exists in the database
                if len(id) == 0:        # pin not found, can be created
                    msg = "Account register succesfull, please say 0 for login and 1 for another register"
                    current_id = new_doc["_id"]     # Take the ID of the database
                    last_key = list(new_doc.keys())[-1] # Take the key value from a dict
                    last_key_number = take_current_pin_number(last_key) + 1 # Take the integer from the PIN ID                   
                    filter_criteria = {"_id": current_id}
                    # Define the update operation (set a new value for the "pin" field)
                    dictKeys = "pin" + str(last_key_number)
                    new_doc[dictKeys] = loginToken 
                    update_operation = {"$set": new_doc}    # Update the Database

                    # Update the document
                    result = collection.update_one(filter_criteria, update_operation)
                    cursor = collection.find()
                    for document in cursor:
                        new_doc = document
                    voice_notification(msg)   
                    break
                
                else:
                    msg = "Account already exists, please try again"
                    voice_notification(msg)

# Login Function
# 2nd Function to land, will go to main function if login succeeded or keep looping until the pin is correct
def login():
    global db
    global collection
    global tokenList
    db = client['GP']      # Take the cluster from a mongodb Cloud
    collection = db["ob"]       # Take the database from a mongodb Cloud
    cursor = collection.find()      # Find all the database from the collection variable
    new_doc = {}
    # Iterate through the cursor to take all the database into a dictionary
    for document in cursor:
        new_doc = document      
    mic = WhisperMic(english=True,model="tiny",energy=1000)     # Start the mic, with the tiny version of the model and the energy to 1000 (the higher it is, the less sensitive the mic from a noise)
    text = "Please enter a 7 digit number"
    voice_notification(text)
    print("Please enter a 7 digit number...")   
    while True:
        tokenList = []
        mic.energy = 1500       # Set it into 1500 to make sure noise is minimum
        text = mic.listen(timeout=1)           # Listen into the mic, give a timeout of 1 second
        text = text.lower()                    # Change all of the Speech to text string into lowercases 
        if ("timeout" not in text):
            # Remove all of the special characters from the detected text
            text = remove_special_character(text,".")
            text = remove_special_character(text,",")
            text = remove_special_character(text,"!")
            print("Input Text:" + text)
            if ("three" in text or "3" in text ):  # If three or 3 input is detected, go to signup
                new_text = remove_special_character(text," ")
                if (new_text == "three" or new_text == "3"):
                    signup()
                    break
            text = text.split() # Split the test into an array
            tokenList= speech_to_text(text)     # Change all string output into an integer
            print(tokenList)
            if (len(tokenList) > 7):        # Check if the token is higher than 7
                msg = "Input is higher than 7 digit, please try again"
                voice_notification(msg)
            elif (len(tokenList) < 7):      # Check if the token is less than 7
                msg = "Input is lower than 7 digit, please try again"
                voice_notification(msg)
            else:
                loginToken = "".join(tokenList)     # Join all arrays into a string
                id = get_keys_by_value(new_doc, loginToken) # Check if the token exists in the database
                if len(id) != 0:
                    msg = "Login Succesful, Welcome"
                    id = id[0] 
                    print(id)
                    voice_notification(msg)   
                    main()
                else:
                    msg = "Account not found, please try again or say three for signup"
                    voice_notification(msg)
                    

# Start Function
# 1st Function to land, will go to either Login Function or Signup Function depending of the input
def start():    
    mic = WhisperMic(english=True,model="tiny",energy=1000)
    time.sleep(2)

    text = "Welcome, please say 0 for Login and 1 for SignUp"
    voice_notification(text)
    while True: 
        print("Please speak something...")   
        mic.energy = 1500
        text = mic.listen(timeout=1)
        text = text.lower()
        if ("timeout" not in text):
            # Remove all special characters
            text = remove_special_character(text,".")
            text = remove_special_character(text," ")
            text = remove_special_character(text,",")
            text = remove_special_character(text,"!")

            print(text)
            print(len(text))

            if (text == "zero" or text == "0"):
                login()
            elif (text == "one" or text == "1"):
                signup()
            else:
                msg = "is not a command"
                voice_notification(str(text + msg))

# Main Function
# 3rd Function to land, will run voice_command and video_to_images function simultaneously
def main():
    video_file = 0  
    voice_thread = threading.Thread(target=voice_command)
    
    # Start the thread for the voice recognition
    voice_thread.start() 
    video_to_images(video_file)

if __name__ == "__main__":
    start()

