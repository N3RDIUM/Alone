import speech_recognition as sr
import gpt4all
import gtts
import playsound
import os

r = sr.Recognizer()
m = sr.Microphone()

class SpeechEngine:
    def __init__(self, rate=100, volume=0.5):
        self.rate = rate
        self.volume = volume
        print("A moment of silence, please...")
        with m as source: r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))

    def listen(self):
        print("Say something!")
        with m as source: audio = r.listen(source)
        print("Got it! Now to recognize it...")
        try:
            # recognize speech using OpenAI's Speech Recognition
            value = r.recognize_whisper(audio)
            return value
        except sr.UnknownValueError:
            print("Oops! Didn't catch that")
            return ""
        
    def speak(self, text):
        try:
            gtt = gtts.gTTS(text, lang="en", slow=False)
            gtt.save("temp.mp3")
            playsound.playsound("temp.mp3")
            os.remove("temp.mp3")
        except:
            pass
            
class ChatEngine(gpt4all.GPT4All):
    def __init__(self):
        super().__init__("ggml-gpt4all-j-v1.3-groovy")
        self.conversation = []
    
    def get_response(self, prompt):
        return self.generate(prompt)
    
    def generate(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})
        response = super().chat_completion(self.conversation, verbose=False)["choices"][0]["message"]
        self.conversation.append(response)
        return response["content"]
        
if __name__ == "__main__":
    speech = SpeechEngine()
    chat = ChatEngine()
    speech.speak("Hello, I am Alone.")
    while True:
        text = speech.listen()
        print("You: " + text)
        if text is not None:
            response = chat.get_response(str(text))
            print(f"Bot: " + response)
            speech.speak(response)