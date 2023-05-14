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
        print("\rSet minimum energy threshold to {}".format(r.energy_threshold))

    def listen(self, whisper=True):
        print("Say something!")
        with m as source: audio = r.listen(source)
        print("\rGot it! Now to recognize it...")
        try:
            # recognize speech using OpenAI's Speech Recognition
            if whisper:
                value = r.recognize_whisper(audio)
            else:
                try:
                    value = r.recognize_google(audio)
                except:
                    value = r.recognize_whisper(audio)
            return value
        except sr.UnknownValueError:
            print("\rOops! Didn't catch that")
            return ""
        except:
            pass
        
    def speak(self, text):
        try:
            gtt = gtts.gTTS(text,)
            gtt.save("temp.mp3")
            playsound.playsound("temp.mp3")
            os.remove("temp.mp3")
        except:
            pass
            
class ChatEngine(gpt4all.GPT4All):
    def __init__(self):
        super().__init__("ggml-gpt4all-j-v1.3-groovy")
        self.conversation = [
            { "role": "system", "content": "Hello, I am Alone, a speech chatbot." },
            { "role": "system", "content": """### Instruction:
            1. You are a speech chatbot. So respond like one.
            2. You have to speak in English only.
            3. Generate your response yourself. Don't continue the user's sentence. Respond like a human.
            4. As you are a SPEECH chatbot, don't use markdown, html, backticks, etc. 
            Example: Don't use something like this: `Hello, I am a chatbot.` or ###
            5. The prompt below is a question to answer, a task to complete, or a conversation 
            to respond to; decide which and write an appropriate response.
            \n### Prompt: 
            """}
        ]
    
    def get_response(self, prompt):
        return self.generate(prompt)
    
    def generate(self, prompt):
        self.conversation.append({"role": "user", "content": prompt + "\nResponse:"})
        response = super().chat_completion(
            self.conversation, verbose=False, 
            default_prompt_header=False,
            default_prompt_footer=False,
        )["choices"][0]["message"]
        self.conversation.append(response)
        return response["content"]
        
if __name__ == "__main__":
    speech = SpeechEngine()
    chat = ChatEngine()
    speech.speak("Hello, I am Alone, a speech chatbot.")
    while True:
        text = speech.listen()
        print("You: " + text)
        if text is not None and text != "":
            response = chat.get_response(str(text))
            print(f"Bot: " + response)
            speech.speak(response)