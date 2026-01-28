# Qwen3-TTS ローカル音声生成プロジェクト計画

最新の音声生成モデル **Qwen3-TTS** を用いて、ローカル環境で高品質な日本語音声を生成するシステムを構築します。
Docker を使用して環境を構築し、Web UI (Gradio) で簡単に操作できるようにします。

## 1. 要件定義

### 必須要件
- **利用モデル**: Qwen3-TTS (1.7B モデル)
- **実行環境**:
    - OS: Windows (Docker Desktop)
    - GPU: NVIDIA RTX 5060 Ti (VRAM 16GB) - CUDA 12.x
- **インターフェース**: Web UI (Gradio)
    - ブラウザからテキスト入力 -> 音声生成・再生・保存
- **入力**: 日本語テキスト
- **出力**: 音声ファイル (.wav)

## 2. 開発ステップ

1. **Docker環境構築**
    - `Dockerfile` 作成 (Base: PyTorch/CUDA image)
    - `docker-compose.yml` 作成 (GPUパススルー設定)
    - 必要なライブラリ選定: `torch`, `transformers`, `accelerate`, `qwen-tts`, `gradio`, `flash-attn`
2. **アプリケーション実装**
    - `app.py`: Gradio を使用したUIと推論ロジック
    - CLIからの簡易実行テスト (Optional)
3. **動作検証**
    - Docker コンテナのビルドと起動
    - ブラウザでの音声生成テスト 