# Text Summarization using T5 Transformer

A modern web application for automatic text summarization powered by the T5 (Text-to-Text Transfer Transformer) model from Hugging Face.

## Features

✨ **Advanced NLP**: Uses pre-trained T5 model for high-quality text summarization  
🎨 **Modern UI**: Clean, responsive web interface with smooth interactions  
⚡ **Fast Processing**: Efficient text summarization with GPU/CPU support  
🔧 **Easy Setup**: Simple installation with requirements.txt  

## Tech Stack

- **Backend**: FastAPI + Python
- **Frontend**: HTML, CSS, JavaScript
- **NLP Model**: T5 Transformer (Hugging Face)
- **Server**: Uvicorn
- **Deep Learning**: PyTorch, Transformers

## Prerequisites

- Python 3.8+
- pip or conda

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DivyanshRajSoni/Text_Summarization_using_T5.git
   cd Text_Summarization_using_T5
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the T5 model**
   The model will be automatically downloaded on first run from Hugging Face.

## Usage

1. **Start the server**
   ```bash
   python app.py
   ```
   or
   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Open in browser**
   - Navigate to `http://127.0.0.1:8000`
   - Paste your text in the textarea
   - Click "Summarize" to get the summary

## Project Structure

```
.
├── app.py                    # FastAPI application & API endpoints
├── index.html               # Frontend HTML/CSS/JavaScript
├── requirements.txt         # Python dependencies
├── saved_summary_model/     # T5 model files (auto-downloaded)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── README.md               # This file
```

## API Endpoints

### POST `/summarize/`
Summarizes the provided text.

**Request:**
```json
{
  "dialogue": "Your long text here..."
}
```

**Response:**
```json
{
  "summary": "Summarized text..."
}
```

### GET `/`
Returns the web interface.

## Configuration

- **Max input length**: 512 tokens
- **Max summary length**: 150 tokens
- **Number of beams**: 4 (for beam search)
- **Device**: Auto-detects GPU (CUDA/MPS) or uses CPU

## Requirements

See [requirements.txt](requirements.txt) for complete dependencies:
- fastapi
- uvicorn
- transformers
- torch
- pydantic

## Performance Tips

- **GPU Support**: Install CUDA-enabled PyTorch for faster processing
- **Model Size**: T5 model (~230 MB) is downloaded on first run
- **Concurrency**: FastAPI handles multiple requests efficiently

## Troubleshooting

**Issue**: Model download fails
- **Solution**: Check internet connection and Hugging Face accessibility

**Issue**: Out of Memory
- **Solution**: Use a smaller model or reduce max_length in app.py

**Issue**: Port 8000 already in use
- **Solution**: Change port: `--port 8001`

## Future Enhancements

- [ ] Support multiple summarization models
- [ ] Batch processing
- [ ] User authentication
- [ ] Summary history
- [ ] Export to PDF/TXT

## Author

**Divyansh Raj Soni**  
GitHub: [@DivyanshRajSoni](https://github.com/DivyanshRajSoni)

## License

MIT License - feel free to use this project!

## References

- [T5 Transformer Paper](https://arxiv.org/abs/1910.10683)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Made with ❤️ using FastAPI & Transformers**
