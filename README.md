---
title: Transcriptocast AI Demo
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: 3.10.0
app_file: app.py
pinned: false
---

# 🎙️ Transcriptocast AI

A powerful AI-powered application that provides **audio transcription**, **text summarization**, and **multi-language translation** capabilities. Built with FastAPI and deployed on Hugging Face Spaces.

## 🌟 Key Features

- **Audio Transcription**: Convert audio to text using OpenAI's Whisper model
- **Text Summarization**: Generate concise summaries using Facebook's BART model
- **Multi-language Translation**: Translate between multiple languages using mBART

## 🚀 Quick Start

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transcriptocast.git
cd transcriptocast
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker Deployment

Build and run using Docker:
```bash
docker build -t transcriptocast .
docker run -p 7860:7860 transcriptocast
```

## 📚 API Documentation

### Endpoints

1. **Transcribe Audio** (`POST /transcribe`)
   - Converts audio files to text
   - Accepts: Audio file (MP3, WAV, etc.)
   - Returns: Transcribed text

2. **Summarize Text** (`POST /summarize`)
   - Generates concise summaries
   - Accepts: Text input
   - Returns: Summary

3. **Translate Text** (`POST /translate`)
   - Translates text between languages
   - Accepts: Text and language codes
   - Returns: Translated text

## 🛠️ Technical Stack

- **Backend Framework**: FastAPI
- **AI Models**:
  - Whisper (OpenAI) for transcription
  - BART (Facebook) for summarization
  - mBART (Facebook) for translation
- **Deployment**: Hugging Face Spaces
- **Container**: Docker

## 🔧 Configuration

The application uses the following environment variables:
- `TRANSFORMERS_CACHE`: Cache directory for models
- `HF_HOME`: Hugging Face home directory

## 📦 Model Information

### Whisper Model
- **Type**: Speech-to-Text
- **Version**: Base
- **Use Case**: Audio transcription
- **Size**: ~1GB

### BART Model
- **Type**: Text Summarization
- **Model**: facebook/bart-large-cnn
- **Use Case**: Text summarization
- **Features**: Abstractive summarization

### mBART Model
- **Type**: Machine Translation
- **Model**: facebook/mbart-large-50-many-to-many-mmt
- **Use Case**: Multi-language translation
- **Languages**: 50+ languages

## 🌐 Live Demo

Try the live demo at: [Hugging Face Space](https://huggingface.co/spaces/Prashant26am/transcriptocast-demo)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Contact

For any questions or suggestions, please open an issue in the repository.

---
Made with ❤️ by Prashant Ambati

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference 
