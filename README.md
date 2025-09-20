
# AI Voice Assistant (Web) — Python (Flask)

**What this is**
- A fully functional *starter* AI Voice Assistant implemented as a Flask web app.
- Local speech-to-text using VOSK (if you install a model) with fallback to SpeechRecognition/Google API.
- Text intent classification using a small scikit-learn pipeline (TF-IDF + LogisticRegression). You can retrain with your own diversified dataset.
- Text-to-speech using `pyttsx3` (works offline).
- Frontend records audio in the browser and sends to the backend for processing.

**Important notes**
- This repository provides a complete, runnable app for **Python 3.10+** and VS Code.
- Heavy model training (large neural networks) and downloading large pretrained speech models (VOSK, transformers) are **NOT** performed here — those require internet and substantial compute. This project provides the training scripts and clear instructions to use diversified datasets and stricter parameters for training on your machine or a cloud instance.
- For the strongest offline STT, install a VOSK model and set `VOSK_MODEL_PATH` in the README instructions.

## How to run (quick)
1. Create a Python 3.10+ virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate         # Windows (PowerShell)
   pip install -r requirements.txt
   ```
2. (Optional but recommended) Download a VOSK small model (e.g., `vosk-model-small-en-us-0.15`) and place it in `models/vosk-model`.
   Set environment variable `VOSK_MODEL_PATH=models/vosk-model`.
3. Train the intent model (optional):
   ```bash
   python train.py
   ```
   This will create `models/intent_model.joblib`.
4. Run the Flask app:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000` in your browser. Click **Start Recording**, speak, and the assistant will transcribe and respond.

## Files of interest
- `app.py` — Flask web server and endpoints.
- `assistant.py` — NLU (intent classifier), TTS helper, and processing pipeline.
- `train.py` — Simple training script using a sample diversified dataset (`dataset.csv`). Replace with your dataset for stricter training.
- `templates/index.html` and `static/script.js` — Frontend UI and recorder.
- `requirements.txt` — Python dependencies.

## Custom training for stricter parameters
- Replace `dataset.csv` with a larger, well-labeled diversified dataset (many speakers, accents, noise conditions).
- In `train.py`, change the model (e.g., use neural classifiers, tune hyperparameters, use cross-validation and class weighting).
- Consider using transformer-based models (Hugging Face) for intent classification if you have GPU and internet.

Enjoy! — Generated automatically by ChatGPT.
