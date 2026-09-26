from gtts import gTTS
import os

# Read the text from the .txt file
with open('sentence.txt', 'r') as file:
    text = file.read()

# Convert the text to speech
tts = gTTS(text=text, lang='en')

# Save the audio file
tts.save('output.mp3')

# Optionally, play the audio file
# os.system('start output.mp3')  # For Windows
os.system('afplay output.mp3')  # For MacOS
# os.system('mpg321 output.mp3')  # For Linux
