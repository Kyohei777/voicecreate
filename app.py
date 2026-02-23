import gradio as gr
import subprocess
import torch
import logging
import os
import numpy as np
import soundfile as sf
import librosa
import pickle
import glob
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from qwen_tts import Qwen3TTSModel
from faster_whisper import WhisperModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VOICES_DIR = "saved_voices"
os.makedirs(VOICES_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
whisper_model = None
llm_model = None
llm_tokenizer = None

# Runtime State
cached_clone_prompt = None
cached_ref_audio_path = None
# profile_mode: False = standard/zero-shot, True = using deep profile
current_profile_data = None 

def load_model():
    global model, whisper_model, llm_model, llm_tokenizer
    if model is not None:
        return "✅ All models already loaded!"
    
    try:
        logger.info(f"Loading Qwen3-TTS: {MODEL_ID}...")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map="auto"
        )
        
        logger.info("Loading Whisper (large-v3)...")
        whisper_model = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "int8")
        
        logger.info(f"Loading LLM: {LLM_MODEL_ID} (8-bit quantized)...")
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True if device=="cuda" else False,
            low_cpu_mem_usage=True
        )
        
        return "✅ All models loaded successfully!"
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return f"❌ Error: {e}"

def convert_video_to_audio(video_path):
    if video_path is None:
        return None, "⚠️ No video uploaded"
    output_path = os.path.splitext(video_path)[0] + ".wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", output_path], check=True)
        return output_path, "✅ Audio extracted!"
    except Exception as e:
        logger.error(f"Error: {e}")
        return None, f"❌ Error: {e}"

def transcribe_audio(audio_file):
    global whisper_model
    if whisper_model is None:
        return "⚠️ Please load models first"
    if audio_file is None:
        return "⚠️ Please upload audio first"

    try:
        segments, info = whisper_model.transcribe(audio_file, beam_size=5, language="ja")
        text = "".join([seg.text for seg in segments])
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"❌ Error: {e}"

def correct_transcription(raw_text):
    global llm_model, llm_tokenizer
    
    if llm_model is None:
        return "⚠️ Please load models first"
    if not raw_text or raw_text.strip() == "":
        return "⚠️ No text to correct"
    
    try:
        prompt = f"""以下の音声書き起こしを校正してください。
誤字脱字を修正し、適切な句読点を追加してください。
【重要】話者のイントネーションやリズムを維持するため、「えーと」「あの」などのフィラーや言い淀みは削除せずに、聞こえた通りに残してください。
校正後のテキストのみを出力してください。

入力: {raw_text}

出力:"""

        messages = [{"role": "user", "content": prompt}]
        text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = llm_tokenizer(text, return_tensors="pt").to(llm_model.device)
        
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True, pad_token_id=llm_tokenizer.eos_token_id)
        
        response = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        logger.error(f"Correction error: {e}")
        return f"❌ Error: {e}"

def cache_speaker_embedding(ref_audio, ref_text):
    global model, cached_clone_prompt, cached_ref_audio_path, current_profile_data
    
    if model is None:
        return "⚠️ Please load models first"
    if not ref_audio:
        return "⚠️ Please upload audio first"
    
    try:
        logger.info(f"Creating voice clone prompt...")
        prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text if ref_text else None)
        
        # Zero-shot mode update
        cached_clone_prompt = prompt
        cached_ref_audio_path = ref_audio
        current_profile_data = None # Reset deep profile
        
        return f"✅ Voice cached! (Standard Mode)"
    except Exception as e:
        logger.error(f"Cache error: {e}")
        return f"⚠️ Fallback mode: {e}"

# --- Deep Profiling Logic with Smart Selection ---

# Style keywords for fast matching (no LLM needed at generation time)
STYLE_KEYWORDS = {
    "question": ["？", "?", "か？", "ですか", "なの", "どう", "何", "いつ", "どこ", "誰", "なぜ"],
    "excited": ["！", "!", "すごい", "やった", "最高", "嬉しい", "楽しい", "ワクワク"],
    "sad": ["悲しい", "辛い", "残念", "寂しい", "泣", "切ない"],
    "angry": ["怒", "ふざけ", "許さない", "ひどい", "最悪"],
    "greeting": ["こんにちは", "おはよう", "こんばんは", "はじめまして", "どうも"],
    "calm": ["です。", "ます。", "思います", "考えます", "と言えます"],
}

def calculate_snr(audio_segment, sr=24000):
    """Calculate Signal-to-Noise Ratio (simplified RMS-based)."""
    rms = np.sqrt(np.mean(audio_segment**2))
    # Estimate noise from quietest 10% of the signal
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    frames = librosa.util.frame(audio_segment, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames**2, axis=0))
    noise_rms = np.percentile(frame_rms, 10)
    if noise_rms < 1e-10:
        noise_rms = 1e-10
    snr = 20 * np.log10(rms / noise_rms)
    return snr

def is_complete_sentence(text):
    """Check if text ends with proper punctuation."""
    text = text.strip()
    return text.endswith(("。", "？", "！", ".", "?", "!", "…"))

def detect_style_fast(text):
    """Fast keyword-based style detection (no LLM)."""
    for style, keywords in STYLE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return style
    return "neutral"

def extract_audio_prosody(audio_segment, sr=24000):
    """Extract prosody features directly from audio (pitch, tempo, energy)."""
    try:
        # Pitch analysis using librosa
        pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_range = max(pitch_values) - min(pitch_values)
        else:
            pitch_mean, pitch_std, pitch_range = 200.0, 50.0, 100.0
        
        # Tempo analysis
        tempo, _ = librosa.beat.beat_track(y=audio_segment, sr=sr)
        if hasattr(tempo, '__iter__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        
        # Energy analysis
        rms = librosa.feature.rms(y=audio_segment)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        
        # Duration
        duration = len(audio_segment) / sr
        
        # Pitch contour (normalized time series for pattern matching)
        pitch_contour = extract_pitch_contour(audio_segment, sr)
        
        return {
            "pitch_mean": pitch_mean / 500.0,  # Normalize to 0-1 range
            "pitch_std": pitch_std / 200.0,
            "pitch_range": pitch_range / 500.0,
            "tempo": tempo / 200.0,
            "energy_mean": min(energy_mean * 10, 1.0),
            "energy_std": min(energy_std * 10, 1.0),
            "duration": min(duration / 15.0, 1.0),
            "pitch_contour": pitch_contour  # NEW: pitch curve for matching
        }
    except Exception as e:
        logger.warning(f"Audio prosody extraction failed: {e}")
        return {
            "pitch_mean": 0.4, "pitch_std": 0.25, "pitch_range": 0.2,
            "tempo": 0.6, "energy_mean": 0.3, "energy_std": 0.1, "duration": 0.5,
            "pitch_contour": None
        }

def extract_pitch_contour(audio_segment, sr=24000, n_points=20):
    """Extract normalized pitch contour as a fixed-length vector."""
    try:
        # Use pyin for more robust pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_segment, fmin=50, fmax=500, sr=sr
        )
        
        # Replace NaN with interpolated values
        f0_clean = np.copy(f0)
        nans = np.isnan(f0_clean)
        if np.any(~nans):
            f0_clean[nans] = np.interp(
                np.flatnonzero(nans), 
                np.flatnonzero(~nans), 
                f0_clean[~nans]
            )
        else:
            return None
        
        # Resample to fixed length
        indices = np.linspace(0, len(f0_clean) - 1, n_points).astype(int)
        contour = f0_clean[indices]
        
        # Normalize to 0-1 range
        contour_min = np.min(contour)
        contour_max = np.max(contour)
        if contour_max > contour_min:
            contour = (contour - contour_min) / (contour_max - contour_min)
        else:
            contour = np.zeros(n_points) + 0.5
        
        return contour.tolist()
    except Exception:
        return None

def calculate_pitch_contour_similarity(contour1, contour2):
    """Calculate similarity between two pitch contours using correlation."""
    if contour1 is None or contour2 is None:
        return 0.5  # Default medium similarity
    
    try:
        c1 = np.array(contour1)
        c2 = np.array(contour2)
        
        # Ensure same length
        if len(c1) != len(c2):
            min_len = min(len(c1), len(c2))
            c1 = c1[:min_len]
            c2 = c2[:min_len]
        
        # Calculate correlation coefficient
        if np.std(c1) > 0 and np.std(c2) > 0:
            correlation = np.corrcoef(c1, c2)[0, 1]
            # Convert to 0-1 range (correlation is -1 to 1)
            similarity = (correlation + 1) / 2
        else:
            similarity = 0.5
        
        return float(similarity)
    except Exception:
        return 0.5

def extract_prosody_features(text):
    """Extract prosody-related features from text for similarity matching."""
    text = text.strip()
    
    # Feature 1: Text length (character count, normalized)
    length = len(text)
    length_normalized = min(length / 100.0, 1.0)  # Normalize to 0-1
    
    # Feature 2: Punctuation density
    punct_count = text.count("、") + text.count("。") + text.count("，") + text.count(".") + text.count(",")
    punct_density = punct_count / max(length, 1)
    
    # Feature 3: Question marker (strong prosody indicator)
    is_question = 1.0 if ("？" in text or "?" in text or text.endswith("か。") or text.endswith("の。")) else 0.0
    
    # Feature 4: Exclamation marker
    is_exclaim = 1.0 if ("！" in text or "!" in text) else 0.0
    
    # Feature 5: Emotional intensity (keyword-based score)
    emotion_keywords = ["すごい", "本当に", "とても", "非常に", "絶対", "めちゃ", "超", "マジ"]
    emotion_score = sum(1 for kw in emotion_keywords if kw in text) / len(emotion_keywords)
    
    # Feature 6: Sentence count (rhythm indicator)
    sentence_count = text.count("。") + text.count("！") + text.count("？") 
    sentence_normalized = min(sentence_count / 5.0, 1.0)
    
    return {
        "length": length_normalized,
        "punct_density": punct_density,
        "is_question": is_question,
        "is_exclaim": is_exclaim,
        "emotion": emotion_score,
        "sentences": sentence_normalized
    }

def calculate_prosody_similarity(features1, features2):
    """Calculate similarity between two prosody feature vectors (0-1, higher is better)."""
    # Weighted Euclidean distance
    weights = {
        "length": 1.0,
        "punct_density": 1.5,
        "is_question": 3.0,  # High weight for question intonation
        "is_exclaim": 2.5,
        "emotion": 1.5,
        "sentences": 1.0
    }
    
    total_weight = sum(weights.values())
    distance = 0.0
    
    for key in weights:
        diff = features1.get(key, 0) - features2.get(key, 0)
        distance += weights[key] * (diff ** 2)
    
    distance = np.sqrt(distance / total_weight)
    # Convert distance to similarity (0-1)
    similarity = 1.0 / (1.0 + distance)
    return similarity

def analyze_style(text):
    """Analyze the style/emotion of a text segment using LLM."""
    global llm_model, llm_tokenizer
    if not text: return "neutral"
    
    try:
        prompt = f"""以下の文章の感情や口調を短いキーワード（タグ）で分類してください。
例: [喜び], [怒り], [悲しみ], [疑問], [断定], [冷静], [興奮], [挨拶]
出力はタグのみにしてください。

文章: {text}
タグ:"""
        
        messages = [{"role": "user", "content": prompt}]
        text_in = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = llm_tokenizer(text_in, return_tensors="pt").to(llm_model.device)
        with torch.no_grad():
             outputs = llm_model.generate(**inputs, max_new_tokens=20, temperature=0.1, pad_token_id=llm_tokenizer.eos_token_id)
        tag = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return tag
    except:
        return "neutral"

def analyze_full_video(video_path, profile_name, progress=gr.Progress()):
    global model
    if not video_path:
        yield "⚠️ No vide selected", []
        return
    if not profile_name:
        yield "⚠️ Enter a name for the profile", []
        return
    if model is None:
        yield "⚠️ Load models first", []
        return

    # 1. Extract Audio
    progress(0.1, desc="Extracting audio...")
    yield "⏳ Step 1/5: Extracting audio...", []
    audio_path, status = convert_video_to_audio(video_path)
    if not audio_path:
        yield status, []
        return

    # 2. Segment using Librosa (Silence Based) with Quality Filtering
    progress(0.3, desc="Segmenting audio...")
    yield "⏳ Step 2/5: Segmenting & filtering audio...", []
    try:
        y, sr = librosa.load(audio_path, sr=24000)
        # Slightly looser split to get more variety
        intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
        
        valid_segments = []
        for start_idx, end_idx in intervals:
            duration = (end_idx - start_idx) / sr
            # RELAXED: 2-15 seconds (was 3-10)
            if 2.0 <= duration <= 15.0:
                seg_audio = y[start_idx:end_idx]
                # SNR threshold: 2dB
                snr = calculate_snr(seg_audio, sr)
                if snr >= 2.0:
                    valid_segments.append((start_idx, end_idx, snr))
        
        # Sort by SNR (best quality first), keep top 15
        valid_segments.sort(key=lambda x: x[2], reverse=True)
        target_segments = [(s, e) for s, e, _ in valid_segments[:30]]
        
        if not target_segments:
            yield "❌ No high-quality segments found. Try a cleaner audio source.", []
            return
            
        logger.info(f"Found {len(target_segments)} high-quality segments (SNR >= 2dB)")
            
    except Exception as e:
        yield f"❌ Segmentation error: {e}", []
        return

    # 3. Analyze each segment
    profile_prompts = []
    
    for i, (start, end) in enumerate(target_segments):
        progress(0.3 + (0.6 * (i/len(target_segments))), desc=f"Analyzing segment {i+1}/{len(target_segments)}...")
        yield f"⏳ Step 3/5: Analyzing segment {i+1}...", []
        
        seg_audio = y[start:end]
        # Clean up prompt audio (remove silence/noise at edges)
        seg_audio = cleanup_voice_prompt(seg_audio, sr)

        # Memory-based processing roughly, but Qwen needs file path sometimes.
        # We will overwrite the same temp file to save space? No, parallel needs distinct.
        # But we are serial here.
        seg_path = "/tmp/temp_segment.wav" 
        sf.write(seg_path, seg_audio, sr)
        
        try:
           # Transcribe
           raw_text = transcribe_audio(seg_path)
           corrected = correct_transcription(raw_text)
           if "Error" in corrected or len(corrected) < 2: continue
           
           # Sentence completeness check - RELAXED: allow incomplete for more diversity
           # (Previously required sentences to end with 。？！)
           
           # Style Tagging using LLM (only during profile creation)
           style_tag_llm = analyze_style(corrected)
           # Also store fast-match style for generation time
           style_tag_fast = detect_style_fast(corrected)
           
           # Generate Prompt
           prompt = model.create_voice_clone_prompt(ref_audio=seg_path, ref_text=corrected)
           
           profile_prompts.append({
               "text": corrected,
               "style": style_tag_llm,
               "style_fast": style_tag_fast,
               "prosody": extract_prosody_features(corrected),
               "audio_prosody": extract_audio_prosody(seg_audio, sr),
               "prompt": prompt,
               "score": len(seg_audio)
           })
           
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")
            continue

    if not profile_prompts:
        yield "❌ Failed to generate any prompts.", []
        return

    # 4. Save Profile
    progress(0.95, desc="Saving Profile...")
    yield "⏳ Step 4/5: Saving Profile...", []
    save_path = os.path.join(VOICES_DIR, f"{profile_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(profile_prompts, f)
        
    progress(1.0, desc="Done!")
    yield f"✅ Profile '{profile_name}' saved! ({len(profile_prompts)} patterns analyzed)", update_voice_list()

def find_best_match(target_text, profile_data):
    """Find the best matching prompt using text + audio prosody similarity."""
    if not profile_data: 
        return None, "No profile"
    
    # Check if profile has prosody data
    has_prosody = "prosody" in profile_data[0] if profile_data else False
    has_audio_prosody = "audio_prosody" in profile_data[0] if profile_data else False
    
    if not has_prosody:
        # Fallback to random for old profiles without prosody data
        import random
        selected = random.choice(profile_data)
        return selected, f"🎲 Random (Old Profile) | Style: {selected.get('style', 'unknown')}"
    
    # Extract target features (safe - no side effects)
    try:
        target_features = extract_prosody_features(target_text)
        target_style = detect_style_fast(target_text)
    except Exception as e:
        import random
        selected = random.choice(profile_data)
        return selected, f"🎲 Random (Feature Error) | {e}"
    
    # Calculate similarity for each profile segment
    scored_items = []
    
    for p in profile_data:
        try:
            # --- Quality Check (Anti-Hallucination) ---
            # Check character rate (characters per second)
            # Normal Japanese speech is ~10-20 chars/sec? No, closer to 5-10 chars/sec.
            # If audio is long but text is short, it likely contains silence or extra speech.
            p_text = p.get("text", "")
            p_audio = p.get("audio") # Not available here usually? prompt is here.
            # We can rely on stored features if available, or approximate.
            # Actually, let's use the 'prosody' duration if available.
            p_prosody = p.get("prosody", {})
            
            # Simple heuristic: If text is very short (< 5 chars) but tagged as high quality, be careful.
            if len(p_text) < 4: 
                continue

            # --- NG Word Filter (Anti-Hallucination) ---
            # If reference text contains problematic phrases, skip it.
            # Ideally this should be done during profiling, but here is safer for existing profiles.
            ng_words = ["が求められています", "ができます", "思いますか", "そうですね"]
            if any(ng in p_text for ng in ng_words):
                 # Only skip if the target text DOES NOT contain these words
                 # If target has them, it's fine to use them.
                 target_has_word = any(ng in target_text for ng in ng_words)
                 if not target_has_word:
                     continue

            profile_features = p.get("prosody", {})
            if not profile_features:
                continue
            
            # Text prosody similarity (primary)
            text_sim = calculate_prosody_similarity(target_features, profile_features)
            
            # Audio prosody bonus (if available)
            audio_bonus = 0.0
            contour_bonus = 0.0
            if has_audio_prosody and "audio_prosody" in p:
                audio_p = p["audio_prosody"]
                
                # Check Duration vs Text Length Ratio
                # If duration is 5s but text is "はい", ratio is 0.4 chars/sec (Suspicious)
                duration = audio_p.get("duration", 0) * 15.0 # Denormalize (approx)
                if duration > 0:
                    char_rate = len(p_text) / duration
                    # Threshold: Less than 2 chars/sec is very suspicious for Japanese
                    if char_rate < 2.0:
                        continue 
                
                # Prefer segments with moderate energy and varied pitch
                energy_score = 1.0 - abs(audio_p.get("energy_mean", 0.3) - 0.4)
                pitch_var_score = min(audio_p.get("pitch_std", 0.1) * 2, 0.5)
                audio_bonus = (energy_score + pitch_var_score) * 0.1
                
                # Pitch contour similarity bonus (NEW)
                profile_contour = audio_p.get("pitch_contour")
                if profile_contour:
                    # Match question intonation (rising) or statement (falling)
                    is_rising = profile_contour[-1] > profile_contour[0] if len(profile_contour) > 1 else False
                    target_is_question = target_features.get("is_question", False)
                    if is_rising == target_is_question:
                        contour_bonus = 0.15  # Bonus for matching intonation pattern
            
            # Style bonus (secondary)
            style_bonus = 0.2 if p.get("style_fast") == target_style else 0.0
            
            # Combined score
            total_score = text_sim + audio_bonus + contour_bonus + style_bonus
            scored_items.append((p, total_score, text_sim))
            
        except Exception:
            continue
    
    if not scored_items:
        import random
        selected = random.choice(profile_data)
        return selected, f"🎲 Random (No Match) | Style: {selected.get('style', 'unknown')}"
    
    # Sort by score, get best match
    scored_items.sort(key=lambda x: x[1], reverse=True)
    best_item, best_score, best_sim = scored_items[0]
    
    # Format match info
    match_info = f"🎯 Match: {best_sim:.2f}"
    if has_audio_prosody:
        match_info += " +Audio"
    match_info += f" | Style: {best_item.get('style_fast', 'unknown')}"
    if target_features.get('is_question'):
        match_info += " [疑問文]"
    if target_features.get('is_exclaim'):
        match_info += " [感嘆文]"
    
    # Add reference text snippet for debugging
    ref_text_full = best_item.get("text", "")
    match_info += f"\nRef({len(ref_text_full)}moji): {ref_text_full}"
    
    return best_item, match_info



def load_profile(profile_name):
    global current_profile_data, cached_clone_prompt
    if not profile_name: return "⚠️ Select a profile"
    path = os.path.join(VOICES_DIR, f"{profile_name}.pkl")
    if not os.path.exists(path): return "❌ Profile not found"
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        current_profile_data = data
        cached_clone_prompt = None 
        return f"✅ Loaded '{profile_name}' (Smart Selection Ready)"
    except Exception as e:
        return f"❌ Error loading: {e}"

def update_voice_list():
    files = glob.glob(os.path.join(VOICES_DIR, "*.pkl"))
    names = [os.path.basename(f).replace(".pkl", "") for f in files]
    return gr.Dropdown(choices=names, value=names[0] if names else None)

# --- Standard Logic ---

def process_video_upload(video_file):
    if video_file is None:
        yield None, "", "", "⚠️ No file uploaded"
        return
    yield None, "", "", "⏳ Extracting audio..."
    audio_path, status = convert_video_to_audio(video_file)
    if not audio_path:
        yield None, "", "", status
        return
    yield audio_path, "", "", "⏳ Audio extracted. Transcribing..."
    raw_text = transcribe_audio(audio_path)
    yield audio_path, raw_text, "", "⏳ Transcribed. Correcting text..."
    corrected_text = correct_transcription(raw_text)
    yield audio_path, raw_text, corrected_text, "⏳ Text corrected. Caching voice..."
    cache_status = cache_speaker_embedding(audio_path, corrected_text)
    yield audio_path, raw_text, corrected_text, cache_status

    yield audio_path, raw_text, corrected_text, cache_status

def cleanup_voice_prompt(audio_data, sr=24000, threshold_db=30):
    """
    Aggressively clean up voice prompt audio to prevent trailing garbage generation.
    Trims silence from both ends and ensures clean cut.
    """
    try:
        # Use librosa to split non-silent parts
        intervals = librosa.effects.split(audio_data, top_db=threshold_db)
        clean_audio = audio_data
        
        if len(intervals) > 0:
            # Take the range from start of first interval to end of last interval
            start = intervals[0][0]
            end = intervals[-1][1]
            
            # Add a tiny fade-out at the end to prevent clicking
            clean_audio = audio_data[start:end]
            if len(clean_audio) > 1000:
                fade_len = 500
                fade = np.linspace(1.0, 0.0, fade_len)
                clean_audio[-fade_len:] *= fade
        
        # Add silence padding to both ends (Strong Stop Signal)
        # 0.1s at start, 0.2s at end
        pad_start = np.zeros(int(sr * 0.1), dtype=np.float32)
        pad_end = np.zeros(int(sr * 0.2), dtype=np.float32)
        
        return np.concatenate([pad_start, clean_audio, pad_end])

    except Exception as e:
        logger.warning(f"Voice prompt cleanup failed: {e}")
        return audio_data

def adjust_audio_speed(audio_data, sr, speed):
    """Adjust audio playback speed without changing pitch."""
    try:
        # Use librosa's time stretch (faster = higher rate)
        stretched = librosa.effects.time_stretch(audio_data, rate=speed)
        return stretched.astype(np.float32)
    except Exception as e:
        logger.warning(f"Speed adjustment failed: {e}")
        return audio_data

def split_sentences(text):
    """Split text into sentences for per-sentence generation."""
    import re
    # Clean newlines first
    text = text.replace("\n", "").strip()
    
    # Split specifically on strong sentence endings, not commas
    sentences = re.split(r'([。！？]+)', text)
    
    # Recombine sentences with their endings
    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if sentence:
            if i + 1 < len(sentences) and re.match(r'^[。！？]+$', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 1
            result.append(sentence)
        i += 1

    # Chunking: Merge sentences until chunk is >= 80 chars
    merged_result = []
    buffer = ""
    
    for s in result:
        if not s: continue
        buffer += s
        if len(buffer) >= 80:
            merged_result.append(buffer)
            buffer = ""
            
    if buffer:
        if merged_result and len(buffer) < 20: 
            # Append very short remainder to previous chunk
            merged_result[-1] += buffer
        else:
            merged_result.append(buffer)
            
    return merged_result

def generate_voice_clone(ref_audio, ref_text, target_text, speed=1.0):
    global model, cached_clone_prompt, cached_ref_audio_path, current_profile_data
    
    if model is None:
        return None, "⚠️ Please load models first"
    if not target_text:
        return None, "⚠️ Please enter text to generate"
    
    try:
        # Split into sentences for per-sentence generation
        sentences = split_sentences(target_text)
        
        # If only one sentence or no profile, use original logic
        if len(sentences) <= 1 or not current_profile_data:
            return generate_single_segment(ref_audio, ref_text, target_text, speed)
        
        logger.info(f"Sentence-level generation: {len(sentences)} sentences")
        
        # Generate each sentence with its best matching segment
        all_audio = []
        log_parts = []
        sample_rate = 24000
        
        for i, sentence in enumerate(sentences):
            logger.info(f"Generating sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
            
            # Find best match for this sentence
            selected_item, reason = find_best_match(sentence, current_profile_data)
            prompt_to_use = copy.deepcopy(selected_item["prompt"])
            
            # Generate
            result = model.generate_voice_clone(
                text=sentence, 
                voice_clone_prompt=prompt_to_use, 
                language="japanese",
                temperature=0.1,  # Strict mode
                repetition_penalty=1.0,  # Disable penalty to prevent forced hallucinations
                top_p=0.7
            )
            
            # Extract audio data
            if isinstance(result, tuple) and len(result) == 2:
                first, second = result
                if isinstance(second, (int, float)) and second > 1000:
                    sample_rate, audio_data = int(second), first
                else:
                    audio_data = first
            else:
                audio_data = result
            
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            elif hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            all_audio.append(audio_data)
            log_parts.append(f"[{i+1}] {reason[:30]}")
        
        # Concatenate all audio with small gaps
        gap = np.zeros(int(sample_rate * 0.1), dtype=np.float32)  # 100ms gap
        combined = []
        for i, audio in enumerate(all_audio):
            combined.append(audio)
            if i < len(all_audio) - 1:
                combined.append(gap)
        
        final_audio = np.concatenate(combined)
        
        # Apply speed adjustment if not 1.0
        if speed != 1.0 and abs(speed - 1.0) > 0.01:
            final_audio = adjust_audio_speed(final_audio, sample_rate, speed)
        
        # Save
        output_path = "/tmp/generated_voice.wav"
        sf.write(output_path, final_audio, samplerate=sample_rate)
        
        log_msg = f"📝 {len(sentences)} sentences generated:\n" + "\n".join(log_parts[:3])
        if len(log_parts) > 3:
            log_msg += f"\n...+{len(log_parts)-3} more"
        
        return output_path, f"✅ Sentence-level complete!\n{log_msg}"
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None, f"❌ Error: {e}"

def generate_single_segment(ref_audio, ref_text, target_text, speed=1.0):
    """Original single-segment generation logic."""
    global model, cached_clone_prompt, cached_ref_audio_path, current_profile_data
    
    try:
        logger.info(f"Generating (single): {target_text[:50]}...")
        
        prompt_to_use = None
        log_msg = ""
        
        if current_profile_data:
            selected_item, reason = find_best_match(target_text, current_profile_data)
            if selected_item and "prompt" in selected_item:
                prompt_to_use = copy.deepcopy(selected_item["prompt"])
                log_msg = f"🧠 Smart Match: {reason}\nRef: {selected_item.get('text', '')[:20]}..."
                logger.info(log_msg)
            else:
                logger.warning("Smart match failed or no prompt found.")
                
        # If smart match failed or not used, try simple cache
        if prompt_to_use is None and cached_clone_prompt is not None:
            prompt_to_use = copy.deepcopy(cached_clone_prompt)
            log_msg = "⚡ Standard Cache Used"

        # If we have a prompt tensor, generate using IT ONLY.
        # DO NOT pass ref_audio to avoid re-transcription by the model library.
        if prompt_to_use:
             result = model.generate_voice_clone(
                text=target_text, 
                voice_clone_prompt=prompt_to_use, 
                language="japanese",
                temperature=0.1,  # Strict mode
                repetition_penalty=1.0, 
                top_p=0.7  # Stricter sampling
            )
        # Only use ref_audio if we DO NOT have a prompt
        elif ref_audio or cached_ref_audio_path:
            audio_to_use = ref_audio if ref_audio else cached_ref_audio_path
            logger.info("Using Direct Audio (This triggers Whisper...)")
            result = model.generate_voice_clone(text=target_text, ref_audio=audio_to_use, ref_text=ref_text if ref_text else None, language="japanese")
            log_msg = "🎵 Direct Audio Used"
        else:
            return None, "⚠️ No voice source selected"
        
        output_path = "/tmp/generated_voice.wav"
        
        if isinstance(result, tuple) and len(result) == 2:
            first, second = result
            if isinstance(second, (int, float)) and second > 1000:
                sample_rate, audio_data = int(second), first
            else:
                sample_rate, audio_data = 24000, first
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            elif hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Apply speed adjustment if not 1.0
            if speed != 1.0 and abs(speed - 1.0) > 0.01:
                audio_data = adjust_audio_speed(audio_data, sample_rate, speed)
            
            sf.write(output_path, audio_data, samplerate=sample_rate)
        else:
            if isinstance(result, list):
                result = np.array(result, dtype=np.float32)
            sf.write(output_path, result, samplerate=24000)
        
        final_msg = f"✅ Generation complete!\n{log_msg}"
        return output_path, final_msg
    except Exception as e:
        logger.error(f"Single generation error: {e}")
        return None, f"❌ Error: {e}"


# Custom CSS
custom_css = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
.main-header { text-align: center; background: linear-gradient(135deg, #FF6B6B 0%, #556270 100%); padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem; color: white; }
.main-header h1 { margin: 0; font-size: 2rem; }
.main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
.tips-accordion { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 1rem; }
.status-box { font-weight: bold; }
.profile-btn { background-color: #f0f0f0; }
"""

# Custom JS
custom_js = """
function() {
    window.setPlaybackRate = function(rate) {
        const audios = document.querySelectorAll('audio');
        audios.forEach(a => a.playbackRate = rate);
        const btns = document.querySelectorAll('.rate-btn');
        btns.forEach(b => b.style.fontWeight = 'normal');
        event.target.style.fontWeight = 'bold';
    }
}
"""

with gr.Blocks(title="Voice Clone Studio Pro", theme=gr.themes.Soft(), css=custom_css, js=custom_js) as demo:
    
    gr.HTML("""
    <div class="main-header">
        <h1>🎙️ Voice Clone Studio Pro</h1>
        <p>Qwen3-TTS | Deep Profiling & Intonation Director</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            status = gr.Textbox(label="System Status", value="🔄 Ready to load models...", interactive=False, elem_classes="status-box")
        with gr.Column(scale=1):
            load_btn = gr.Button("🚀 Load Models", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    with gr.Tabs():
        # --- TAB 1: QUICK CLONE (Original) ---
        with gr.Tab("⚡ Quick Clone (Standard)"):
             with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 1️⃣ Source Material")
                    with gr.Tabs():
                        with gr.Tab("🎬 Video Upload"):
                            ref_video = gr.Video(label="Upload Video (10-30s)", include_audio=True)
                        with gr.Tab("🎵 Audio File"):
                            ref_audio = gr.Audio(label="Reference Audio", type="filepath", interactive=True)
                            transcribe_btn = gr.Button("Transcribe Only", size="sm")

                    gr.Markdown("### 2️⃣ Intonation Tuning")
                    with gr.Accordion("💡 Tips", open=False):
                        gr.Markdown("Best results come from manual filling of fillers like 'umm', 'err'.")
                    
                    with gr.Row():
                         gr.HTML("""<button onclick="setPlaybackRate(0.5)">0.5x</button><button onclick="setPlaybackRate(1.0)">1.0x</button>""")
                    
                    with gr.Row():
                        raw_text = gr.Textbox(label="Raw", lines=2)
                        corrected_text = gr.Textbox(label="Corrected", lines=2)
                    
                    with gr.Row():
                        correct_manual_btn = gr.Button("✨ AI Correct")
                        recache_btn = gr.Button("💾 Re-Cache")
                    process_status = gr.Textbox(label="Log", interactive=False, max_lines=1)

        # --- TAB 2: DEEP PROFILING (New) ---
        with gr.Tab("🧠 Deep Learn (Recommended)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 🏗️ Create Deep Profile
                    Analyze full video (up to 5 mins) to capture speaker's consistent style.
                    """)
                    dp_video = gr.Video(label="Long Video Upload")
                    dp_name = gr.Textbox(label="Profile Name (e.g. 'MyVoice')", placeholder="MyVoice")
                    dp_train_btn = gr.Button("🧠 Start Deep Learning (Profiler)", variant="primary")
                    dp_status = gr.Textbox(label="Progress", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📂 Load Profile
                    Select a trained voice profile to use for generation.
                    """)
                    dp_list = gr.Dropdown(label="Saved Voices", choices=[], interactive=True)
                    dp_refresh = gr.Button("🔄 Refresh List", size="sm")
                    dp_load_btn = gr.Button("📂 Load Selected Voice")
                    dp_load_status = gr.Textbox(show_label=False)

    gr.Markdown("---")
    
    # --- GENERATION SECTION (Shared) ---
    with gr.Row():
        with gr.Column(scale=1):
             gr.Markdown("### 3️⃣ Generate Voice")
             target_text = gr.Textbox(label="Target Text", lines=5)
             with gr.Row():
                 speed_slider = gr.Slider(minimum=0.7, maximum=1.3, value=1.0, step=0.05, label="🚀 Speech Speed")
             generate_btn = gr.Button("🎵 Generate Voice", variant="primary", size="lg")
        with gr.Column(scale=1):
             gr.Markdown("### 🔊 Result")
             output_audio = gr.Audio(label="Output")
             output_status = gr.Textbox(label="Log")

    # Wire Events
    load_btn.click(load_model, outputs=[status])
    
    # Quick Clone
    ref_video.upload(process_video_upload, inputs=[ref_video], outputs=[ref_audio, raw_text, corrected_text, process_status])
    transcribe_btn.click(transcribe_audio, inputs=[ref_audio], outputs=[raw_text])
    correct_manual_btn.click(correct_transcription, inputs=[raw_text], outputs=[corrected_text])
    recache_btn.click(cache_speaker_embedding, inputs=[ref_audio, corrected_text], outputs=[process_status])
    
    # Deep Profile
    dp_train_btn.click(analyze_full_video, inputs=[dp_video, dp_name], outputs=[dp_status, dp_list])
    dp_refresh.click(update_voice_list, outputs=[dp_list])
    dp_load_btn.click(load_profile, inputs=[dp_list], outputs=[dp_load_status])
    
    # Generation (added speed_slider)
    generate_btn.click(generate_voice_clone, inputs=[ref_audio, corrected_text, target_text, speed_slider], outputs=[output_audio, output_status])

    # Init
    def init_ui():
        return update_voice_list()
    
    demo.load(init_ui, outputs=[dp_list])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

