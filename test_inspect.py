import inspect
import sys
try:
    from qwen_tts import Qwen3TTSModel
    print("Qwen3TTSModel found")
    print(inspect.signature(Qwen3TTSModel.from_pretrained))
    print(inspect.signature(Qwen3TTSModel.generate_voice_clone))
except Exception as e:
    print(e)
