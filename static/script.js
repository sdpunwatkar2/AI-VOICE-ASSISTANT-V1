let recordBtn = document.getElementById("record");
let stopBtn = document.getElementById("stop");
let transcriptEl = document.getElementById("transcript");
let replyEl = document.getElementById("reply");

let mediaRecorder;
let audioChunks = [];

recordBtn.onclick = async () => {
  audioChunks = [];
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.start();
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  mediaRecorder.addEventListener("dataavailable", (event) => {
    audioChunks.push(event.data);
  });
};

stopBtn.onclick = async () => {
  mediaRecorder.stop();
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
  // Convert to WAV in the browser using AudioContext
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  // export wav
  function encodeWAV(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const samples = audioBuffer.getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    /* RIFF identifier */
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    // write PCM samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
  }
  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
  const wavBuffer = encodeWAV(audioBuffer);
  const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });
  const reader = new FileReader();
  reader.onloadend = async () => {
    const base64data = reader.result;
    transcriptEl.textContent = "Processing...";
    const resp = await fetch("/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio: base64data }),
    });
    const json = await resp.json();
    transcriptEl.textContent = json.transcript || "—";
    replyEl.textContent = json.reply || "—";

    // Play reply audio if available
    if (json.audio) {
      const audioBytes = atob(json.audio);
      const buffer = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        buffer[i] = audioBytes.charCodeAt(i);
      }
      const blob = new Blob([buffer], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
    }
  };
  reader.readAsDataURL(wavBlob);
};
