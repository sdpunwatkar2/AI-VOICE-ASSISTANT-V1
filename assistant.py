import os
import tempfile
import json
import joblib
import numpy as np
import soundfile as sf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import base64
import io
from gtts import gTTS

# Optional VOSK import handled lazily
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
INTENT_MODEL_PATH = os.path.join(MODELS_DIR, 'intent_model.joblib')
VOSK_MODEL_PATH = os.environ.get('VOSK_MODEL_PATH', '')

class Assistant:
    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
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
                from vosk import Model
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
                import wave
                from vosk import KaldiRecognizer
                wf = wave.open(wav_path, "rb")
                rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
                text_chunks = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text_chunks.append(res.get('text', ''))
                res = json.loads(rec.FinalResult())
                text_chunks.append(res.get('text', ''))
                return ' '.join([t for t in text_chunks if t])
            except Exception as e:
                print("VOSK failed:", e)

        # Fallback: Google Web Speech API
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
            # Use recognize_google method from Recognizer class
            return r.recognize_google(audio)  # type: ignore
        except AttributeError:
            print("speech_recognition.Recognizer does not have 'recognize_google'. Please ensure the library is up to date.")
            return ""
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
            return "I can't fetch live weather in this demo, but you can integrate a weather API."
        # Default echo
        return "You said: " + text

    def synthesize_speech(self, text):
        """Convert reply text to base64 MP3 for web playback"""
        try:
            tts = gTTS(text=text, lang="en")
            mp3_io = io.BytesIO()
            tts.write_to_fp(mp3_io)
            mp3_io.seek(0)
            audio_b64 = base64.b64encode(mp3_io.read()).decode("utf-8")
            return "data:audio/mp3;base64," + audio_b64
        except Exception as e:
            print("TTS failed:", e)
            return ""
