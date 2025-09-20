from flask import Flask, render_template, request, jsonify
import os
from assistant import Assistant
import base64
import io
import soundfile as sf

app = Flask(__name__)
assistant = Assistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    data = request.json
    audio_b64 = data.get('audio')
    if not audio_b64:
        return jsonify({'error': 'no audio provided'}), 400

    # Remove data URL header if present
    header, b64data = audio_b64.split(',', 1) if ',' in audio_b64 else (None, audio_b64)
    audio_bytes = base64.b64decode(b64data)
    audio_file = io.BytesIO(audio_bytes)

    try:
        audio, samplerate = sf.read(audio_file)
    except Exception as e:
        return jsonify({'error': f'could not read audio: {e}'}), 400

    # Run assistant (speech-to-text + reply)
    text, reply = assistant.handle_audio_array(audio, samplerate)

    # Generate TTS (base64 MP3 string)
    reply_audio = assistant.synthesize_speech(reply)

    return jsonify({
        'transcript': text,
        'reply': reply,
        'audio': reply_audio  # base64 MP3
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
