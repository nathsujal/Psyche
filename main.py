import os
import threading
import queue
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ML imports
import joblib
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, ViTImageProcessor, ViTForImageClassification
from groq import Groq
from dotenv import load_dotenv

# -------------------------
# Configuration / Globals
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment. Transcription will fail without it.")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Thread-safe queues
_transcript_queue = queue.Queue()

# -------------------------
# Model loading utility
# -------------------------
def load_all_models():
    """
    Load heavy models once and return them.
    Returns:
        dict containing:
          - 'embedder' : SentenceTransformer
          - 'svm_belong', 'svm_burden' : sklearn-like models loaded from joblib
          - 'tokenizer', 't5_model', 'device'
          - 'vit_processor', 'vit_model'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SentenceTransformer embedder...")
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    print("Loading SVM models (joblib)...")
    svm_belong = joblib.load('svm_belong_mpnet.pkl')
    svm_burden = joblib.load('svm_burden_mpnet.pkl')

    print("Loading FLAN-T5 for explanation generation...")
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    t5_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(device)

    print("Loading ViT face-emotion model...")
    vit_processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
    vit_model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection').to(device)
    vit_model.eval()

    models = {
        'device': device,
        'embedder': embedder,
        'svm_belong': svm_belong,
        'svm_burden': svm_burden,
        'tokenizer': tokenizer,
        't5_model': t5_model,
        'vit_processor': vit_processor,
        'vit_model': vit_model,
    }
    print("All models loaded.")
    return models

# Load models in a background thread so UI can come up fast
_models = {}
_models_loaded_event = threading.Event()

def _background_model_loader():
    global _models
    try:
        _models = load_all_models()
    finally:
        _models_loaded_event.set()

threading.Thread(target=_background_model_loader, daemon=True).start()

# -------------------------
# Helper ML functions
# -------------------------
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_face_emotion(frame_bgr):
    """
    Predict facial emotion for a given OpenCV BGR frame.
    This uses the ViT model loaded in _models.
    Args:
        frame_bgr: numpy array in BGR color (OpenCV standard).
    Returns:
        (label: str, confidence: float)
    """
    if not _models_loaded_event.is_set():
        return "loading...", 0.0

    proc = _models['vit_processor']
    model = _models['vit_model']
    device = _models['device']

    # Convert BGR -> RGB -> PIL Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Preprocess and inference
    inputs = proc(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        idx = torch.argmax(probs, dim=-1).item()
        conf = probs[0][idx].item()

    return EMOTIONS[idx], float(conf)

def predict_labels_from_text(text):
    """
    Given user text, compute mpnet embeddings, then SVM predictions.
    Returns:
        belong_pred (0/1), burden_pred (0/1)
    """
    if not _models_loaded_event.is_set():
        raise RuntimeError("Models are still loading.")

    embedder = _models['embedder']
    svm_belong = _models['svm_belong']
    svm_burden = _models['svm_burden']

    emb = embedder.encode([text], convert_to_numpy=True)
    belong_pred = int(svm_belong.predict(emb)[0])
    burden_pred = int(svm_burden.predict(emb)[0])
    return belong_pred, burden_pred

def generate_explanation(text, belong_pred, burden_pred, max_new_tokens=120):
    """
    Generate a calm, objective explanation using FLAN-T5.
    If both predictions are 0, returns a safe canned message (as in original code).
    """
    if not _models_loaded_event.is_set():
        raise RuntimeError("Models are still loading.")

    tokenizer = _models['tokenizer']
    t5_model = _models['t5_model']
    device = _models['device']

    if belong_pred == 0 and burden_pred == 0:
        return ("The text does not show clear signs of social isolation "
                "or perceived burdensomeness. The content may be explicit "
                "or impulsive, but based on the classifiers, it does not "
                "indicate the two specific psychological states being analyzed.")

    if belong_pred == 1 and burden_pred == 1:
        desc = ("shows combined signals of social isolation "
                "and believing they burden others")
    elif belong_pred == 1:
        desc = "shows signs of social isolation or feeling disconnected"
    elif burden_pred == 1:
        desc = "shows signs of feeling like a burden to others"
    else:
        desc = ("shows no clear indicators of social isolation "
                "or feeling like a burden")

    prompt = f"""
You are analyzing emotional cues in a user's text.
Explain what linguistic or emotional signals indicate the detected mental state.

User text:
\"\"\"{text}\"\"\"

Detected state: The person {desc}.

Write a calm, neutral, and objective explanation based ONLY on:
- emotional tone
- specific phrases
- implications of the wording

Do NOT make assumptions about addiction, diagnosis, or mental disorders.
Do NOT repeat the same idea multiple times.
Do NOT hallucinate or invent facts not present in the text.
"""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# Audio recording utilities
# -------------------------
class Recorder:
    """
    Simple audio recorder using sounddevice.
    Use .start() to start recording to an internal buffer, .stop() to stop,
    and .write_wav(path) to flush buffer to a wav file.
    """
    def __init__(self, samplerate=SAMPLE_RATE, channels=CHANNELS):
        self.samplerate = samplerate
        self.channels = channels
        self._recording = False
        self._frames = []
        self._stream = None

    def _callback(self, indata, frames, time, status):
        # indata is shape (frames, channels)
        if self._recording:
            self._frames.append(indata.copy())

    def start(self):
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(samplerate=self.samplerate,
                                      channels=self.channels,
                                      callback=self._callback)
        self._stream.start()

    def stop(self):
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def write_wav(self, path):
        """
        Concatenate frames and write a WAV file to `path`.
        """
        if not self._frames:
            raise RuntimeError("No audio recorded.")
        audio = np.concatenate(self._frames, axis=0)
        sf.write(path, audio, self.samplerate)
        return path

# -------------------------
# Groq Whisper helper
# -------------------------
def transcribe_with_groq(wav_path):
    """
    Send wav file bytes to Groq Whisper and return the transcript text.
    This blocks until Groq responds.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not available in environment")

    client = Groq(api_key=GROQ_API_KEY)
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(wav_path), f.read()),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json",
        )
    # result may have .text field
    return getattr(result, "text", "") or result.get("text", "")

# -------------------------
# GUI Application (Tkinter)
# -------------------------
class EmotionSpeechApp:
    """
    Main Tkinter application that wires up webcam, audio recorder, and ML pipeline.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion + Speech → Mental-Health Analyzer")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            raise RuntimeError("Could not open webcam.")

        # Recorder instance
        self.recorder = Recorder()

        # UI Elements
        self._build_ui()

        # Video thread controls
        self._stop_video = threading.Event()
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()

    def _build_ui(self):
        """
        Create the UI layout.
        """
        # Top frame: video + emotion
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        # Video canvas
        self.video_label = ttk.Label(top_frame)
        self.video_label.grid(row=0, column=0, rowspan=4, padx=6, pady=6)

        # Live face emotion
        ttk.Label(top_frame, text="Live Face Emotion:").grid(row=0, column=1, sticky=tk.W)
        self.face_emotion_var = tk.StringVar(value="loading...")
        ttk.Label(top_frame, textvariable=self.face_emotion_var, font=("Arial", 12, "bold")).grid(row=0, column=2, sticky=tk.W)

        # Buttons for audio recording
        self.record_btn = ttk.Button(top_frame, text="Start Recording", command=self.start_recording)
        self.record_btn.grid(row=1, column=1, sticky=tk.W, padx=2)
        self.stop_btn = ttk.Button(top_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=2, sticky=tk.W, padx=2)

        # Transcript text box
        ttk.Label(top_frame, text="Transcript:").grid(row=2, column=1, sticky=tk.W, pady=(8,0))
        self.transcript_box = scrolledtext.ScrolledText(top_frame, width=60, height=6, wrap=tk.WORD)
        self.transcript_box.grid(row=3, column=1, columnspan=2, padx=4, pady=4)

        # Predictions and explanation area
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Prediction metrics
        metrics_frame = ttk.Frame(bottom_frame)
        metrics_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(metrics_frame, text="Thwarted Belongingness:").grid(row=0, column=0, sticky=tk.W)
        self.belong_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.belong_var, font=("Arial", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=6)

        ttk.Label(metrics_frame, text="Perceived Burdensomeness:").grid(row=0, column=2, sticky=tk.W)
        self.burden_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.burden_var, font=("Arial", 10, "bold")).grid(row=0, column=3, sticky=tk.W, padx=6)

        # Explanation text box
        ttk.Label(bottom_frame, text="Explanation:").pack(anchor=tk.W, pady=(8,0))
        self.explanation_box = scrolledtext.ScrolledText(bottom_frame, width=100, height=10, wrap=tk.WORD)
        self.explanation_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Analyze button (runs the SVMs + T5 on the current transcript)
        self.analyze_btn = ttk.Button(self.root, text="Analyze Transcript", command=self.run_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=6, pady=6)

        # Small status
        self.status_var = tk.StringVar(value="Models loading in background...")
        ttk.Label(self.root, textvariable=self.status_var).pack(side=tk.RIGHT, padx=6, pady=6)

    def _video_loop(self):
        """
        Runs in background thread: reads frames from webcam, predicts face emotion,
        and updates the UI with the image and live emotion string.
        """
        while not self._stop_video.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 480))

            # Predict face emotion every N frames to save compute (e.g., every 6 frames)
            if not hasattr(self, "_frame_counter"):
                self._frame_counter = 0
            self._frame_counter += 1

            if self._frame_counter % 6 == 0:
                try:
                    label, conf = predict_face_emotion(display_frame)
                except Exception as e:
                    label, conf = "error", 0.0
                    print("Face emotion error:", e)
                self.face_emotion_var.set(f"{label} ({conf:.2f})")

            # Overlay the emotion text on the frame
            cv2.putText(display_frame, self.face_emotion_var.get(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert to PIL Image then ImageTk for tkinter
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            # Update UI label (must be done on main thread; Tkinter is thread-safe for image var if we use .after)
            def _update_image():
                self.video_label.imgtk = imgtk  # keep reference
                self.video_label.config(image=imgtk)

            self.root.after(0, _update_image)

            # small sleep
            cv2.waitKey(1)

        # release camera when loop ends
        self.cap.release()

    # -------------------------
    # Recording controls
    # -------------------------
    def start_recording(self):
        """
        Begin recording audio into Recorder buffer.
        """
        if not _models_loaded_event.is_set():
            messagebox.showinfo("Please wait", "Models are still loading. Try again shortly.")
            return

        try:
            self.recorder.start()
        except Exception as e:
            messagebox.showerror("Recording Error", f"Could not start audio recording: {e}")
            return

        self.record_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Recording...")

    def stop_recording(self):
        """
        Stop recording, write WAV to a temp file, and launch transcription thread.
        """
        self.recorder.stop()
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped recording. Transcribing...")

        # write to temp file
        tmp = tempfile.NamedTemporaryFile(prefix="rec_", suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()
        try:
            self.recorder.write_wav(wav_path)
        except Exception as e:
            messagebox.showerror("Recording Error", f"Failed to write WAV file: {e}")
            return

        # kick off transcription in background
        threading.Thread(target=self._transcribe_and_update, args=(wav_path,), daemon=True).start()

    def _transcribe_and_update(self, wav_path):
        """
        Background thread: send wav to Groq, get transcript and update transcript box.
        """
        try:
            transcript = transcribe_with_groq(wav_path)
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"
            print("Transcription error:", e)

        # Update transcript box on main thread
        def _update_text():
            self.transcript_box.delete(1.0, tk.END)
            self.transcript_box.insert(tk.END, transcript)
            self.status_var.set("Transcription complete. Click 'Analyze Transcript' to run the mental-health models.")
        self.root.after(0, _update_text)

        # cleanup temp file
        try:
            os.remove(wav_path)
        except Exception:
            pass

    def run_analysis(self):
        """
        Take current transcript from UI, run SVMs + T5, and update prediction & explanation areas.
        """
        text = self.transcript_box.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No transcript", "Transcript is empty. Record audio first.")
            return

        self.status_var.set("Running analysis...")
        # run analysis in background
        threading.Thread(target=self._analysis_worker, args=(text,), daemon=True).start()

    def _analysis_worker(self, text):
        """
        Background worker: predicts labels and generates explanation, then updates the UI.
        """
        try:
            belong_pred, burden_pred = predict_labels_from_text(text)
            explanation = generate_explanation(text, belong_pred, burden_pred)
        except Exception as e:
            belong_pred, burden_pred = 0, 0
            explanation = f"[Analysis failed: {e}]"
            print("Analysis error:", e)

        # Update UI on main thread
        def _update_ui():
            self.belong_var.set("YES" if belong_pred else "NO")
            self.burden_var.set("YES" if burden_pred else "NO")
            self.explanation_box.delete(1.0, tk.END)
            self.explanation_box.insert(tk.END, explanation)
            self.status_var.set("Analysis complete.")
        self.root.after(0, _update_ui)

    def _on_close(self):
        """
        Clean up threads and exit cleanly.
        """
        self._stop_video.set()
        try:
            self.recorder.stop()
        except Exception:
            pass
        self.root.destroy()

# -------------------------
# Main entrypoint
# -------------------------
def main():
    root = tk.Tk()
    app = EmotionSpeechApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()