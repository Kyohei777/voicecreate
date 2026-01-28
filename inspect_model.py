import torch
from qwen_tts import Qwen3TTSModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import numpy as np
import soundfile as sf


def inspect_model():
    try:
        model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        
        # Create dummy audio
        dummy_audio = "dummy.wav"
        sf.write(dummy_audio, np.zeros(24000*3), 24000)
        
        logger.info("Creating dummy prompt...")
        prompt = model.create_voice_clone_prompt(ref_audio=dummy_audio)
        logger.info(f"Prompt Type: {type(prompt)}")
        logger.info(f"Prompt Keys: {prompt.keys()}")
        
        for k, v in prompt.items():
            if isinstance(v, torch.Tensor):
                 logger.info(f"Key {k}: Tensor shape {v.shape}")
            elif isinstance(v, list):
                 logger.info(f"Key {k}: List len {len(v)}")
                 if len(v) > 0:
                     logger.info(f"  Element type: {type(v[0])}")
        
        return "Done"
    except Exception as e:
        logger.error(f"Error: {e}")
        return str(e)



if __name__ == "__main__":
    inspect_model()
