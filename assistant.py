
import os
import tempfile
import json
import joblib
import numpy as np
import soundfile as sf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pyttsx3

# Optional VOSK import handled lazily
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

MODELS_DIR = Path = os.path.join(os.path.dirname(__file__), 'models')
INTENT_MODEL_PATH = os.path.join(MODELS_DIR, 'intent_model.joblib')
VOSK_MODEL_PATH = os.environ.get('VOSK_MODEL_PATH', '')

class Assistant:
    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.tts_engine = pyttsx3.init()
        self.intent_clf = None
        if os.path.exists(INTENT_MODEL_PATH):
            try:
                self.intent_clf = joblib.load(INTENT_MODEL_PATH)
            except Exception:
                self.intent_clf = None

        # Load VOSK model if available and path set
        self.vosk_model = None
        if VOSK_AVAILABLE and VOSK_MODEL_PATH and os.path.exists(VOSK_MODEL_PATH):
            try:
                self.vosk_model = Model(VOSK_MODEL_PATH)
            except Exception:
                self.vosk_model = None

    def handle_audio_array(self, audio_array, samplerate):
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_array, samplerate)
            temp_path = f.name
        # Transcribe
        text = self.transcribe(temp_path)
        reply = self.get_reply(text)
        # Speak the reply (non-blocking)
        self.speak(reply)
        # cleanup
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        return text, reply

    def transcribe(self, wav_path):
        # Prefer VOSK offline if available
        if self.vosk_model:
            try:
                wf = open(wav_path, "rb")
                rec = KaldiRecognizer(self.vosk_model, 16000)
                import wave, json
                wf = wave.open(wav_path, "rb")
                rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
                text_chunks = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text_chunks.append(res.get('text',''))
                res = json.loads(rec.FinalResult())
                text_chunks.append(res.get('text',''))
                return ' '.join([t for t in text_chunks if t])
            except Exception as e:
                print("VOSK failed:", e)
        # Fallback: try SpeechRecognition with Google Web Speech API
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
            return r.recognize_google(audio)
        except Exception as e:
            print("Fallback STT failed:", e)
            return ""

    def get_reply(self, text):
        if not text:
            return "Sorry, I didn't catch that. Please try again."
        # Very simple intent classifier
        if self.intent_clf:
            try:
                intent = self.intent_clf.predict([text])[0]
            except Exception:
                intent = None
        else:
            intent = None
        # Rules-based fallback
        text_low = text.lower()
        if 'time' in text_low:
            from datetime import datetime
            return "The current time is " + datetime.now().strftime('%H:%M:%S')
        if 'date' in text_low:
            from datetime import datetime
            return "Today's date is " + datetime.now().strftime('%Y-%m-%d')
        if intent == 'greet':
            return "Hello! How can I help you today?"
        if intent == 'goodbye':
            return "Goodbye! Have a great day."
        if intent == 'weather':
            return "I can't fetch live weather in this offline demo, but you can integrate a weather API."
        # Default echo
        return "You said: " + text

    def speak(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print("TTS failed:", e)
