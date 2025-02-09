# Telegram Image Processing Bot

## Project Overview
This is a Telegram bot that processes and analyzes images. The bot appears to focus on mask detection and classification, likely using machine learning models for image analysis.

## Project Structure
```
telegram-image-internal/
├── data/                  # Data directory (potentially containing model data)
├── model_cache/          # Cached model files
├── test_image/          # Test images directory
├── classify_mask.py     # Mask classification logic
├── config.py           # Configuration settings
├── detection.py        # Detection implementation
├── layers.py          # Neural network layers or processing layers
├── main.py           # Main application entry point
├── mask_identification.py # Mask identification logic
└── requirements.txt   # Project dependencies
```

## Core Components
- **Image Processing Pipeline**: The project implements image processing through multiple stages (preprocessing, detection, classification)
- **Mask Analysis**: Specialized in mask detection and classification
- **Telegram Integration**: Bot interface for handling image submissions and responses

## Key Files
- `main.py`: Entry point and main bot logic
- `classify_mask.py`: Handles mask classification functionality
- `detection.py`: Implements detection algorithms
- `layers.py`: Contains neural network or processing layer definitions
- `mask_identification.py`: Logic for identifying masks in images
- `config.py`: Configuration settings and parameters

## Dependencies
Project dependencies are listed in `requirements.txt`. 

## Deployment
The project is designed to be deployable on PythonAnywhere. Key deployment files:
- `requirements.txt`: Lists all Python package dependencies
- `.env.example`: Template for environment variables

## Development and Debug Features
The project includes debugging capabilities with image output:
- Debug image output (debug_preprocessed_image.png, debug_processed_image.png)
- Test image directory for development

## Environment Setup
1. Create a virtual environment
2. Install dependencies from requirements.txt
3. Configure environment variables based on .env.example

## Notes
- The project uses environment variables for configuration (.env)
- Includes model caching functionality (model_cache directory)
- Contains test suite for deployment verification (test_deployment.py)
- Supports model loading testing (test_model_loading.py)

