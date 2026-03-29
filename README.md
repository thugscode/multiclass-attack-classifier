# Network Intrusion Detection System (IDS)

A machine learning-based Network Intrusion Detection System that classifies network traffic as benign or malicious attacks. This project includes a complete ML pipeline with FastAPI backend and Streamlit web interface for interactive predictions.

## Project Overview

This project implements:
- **Data Preprocessing**: Clean and normalize raw network traffic data from CSV files
- **Model Training**: Train 4 different ML algorithms (Logistic Regression, Random Forest, SVM, Naive Bayes)
- **Model Evaluation**: Comprehensive performance analysis with multiple metrics
- **Inference Pipeline**: Make predictions on new network traffic
- **Web API**: FastAPI REST endpoints for programmatic access
- **Web UI**: Streamlit interface for interactive predictions and visualization

## Dataset

### CICIDS-2017 Dataset
This project uses the **CICIDS-2017** (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset) dataset, which is one of the most comprehensive and realistic network intrusion detection datasets available. 

**Dataset Details:**
- **Source**: [Canadian Institute for Cybersecurity (CIC)](https://www.unb.ca/cic/)
- **Collection Period**: 10 days (Monday to Friday, with multiple weeks of data)
- **Total Samples**: 2.8 million traffic flows
- **Features**: 87 network characteristics extracted from traffic flows
- **Attack Types**: 5 types of attacks plus benign traffic

### Traffic Classification

The system classifies network traffic into 6 categories:
- **Benign**: Normal, non-malicious traffic
- **DoS**: Denial of Service attacks
- **DDoS**: Distributed Denial of Service attacks
- **Probe**: Reconnaissance/probing attacks
- **R2L**: Remote to Local attacks
- **U2R**: User to Root escalation attacks

Each sample contains 87 network features extracted from traffic flows, including:
- Forward and backward packet information (count, size, rate)
- Connection duration and timeout statistics
- TCP flag counts and protocol flags
- Flow entropy and length metrics

## Models Trained

1. **Logistic Regression** - Fast, interpretable linear model
2. **Random Forest** - Ensemble with 100 decision trees
3. **Support Vector Machine (SVM)** - Non-linear kernel (RBF)
4. **Naive Bayes** - Probabilistic classifier

## Project Structure

```
intrusion-detection/
├── data/
│   ├── raw/                      # Original CSV files (10 days of data)
│   └── processed/                # Clean, normalized data + preprocessing objects
├── models/                       # Trained ML models (pickle format)
├── src/
│   ├── data_preprocessing.py     # Clean, normalize, and prepare data
│   ├── train.py                  # Train 4 ML models
│   ├── evaluate.py               # Evaluate model performance
│   ├── predict.py                # Make predictions on new data
│   ├── demo_preprocessing.py     # Demo: 6-row preprocessing walkthrough
│   └── demo_train.py             # Demo: Training with 50 sample rows
├── app/
│   ├── app.py                    # FastAPI backend with REST endpoints
│   ├── streamlit_app.py          # Streamlit web interface
│   └── README.md                 # Web application documentation
├── notebooks/                    # Jupyter notebooks (optional)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thugscode/intrusion-detection.git
cd intrusion-detection
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Data
Clean and normalize raw CSV files:
```bash
cd src
python3 data_preprocessing.py
```

Output files saved to `data/processed/`:
- `{filename}_X.csv` - Features
- `{filename}_y.csv` - Labels
- `{filename}_scaler.pkl` - StandardScaler object
- `{filename}_label_encoder.pkl` - LabelEncoder object

### 2. Train Models
Train all 4 models on preprocessed data:
```bash
python3 train.py
```

Models saved to `models/`:
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`
- `naive_bayes_model.pkl`

### 3. Evaluate Models
Evaluate trained models and compare performance:
```bash
python3 evaluate.py
```

Displays:
- Accuracy, Precision, Recall, F1-Score for each model
- Confusion matrices
- Detailed classification reports

### 4. Make Predictions
Predict on new network traffic data:
```bash
python3 predict.py
```

## Demo Scripts

### See Data Preprocessing Steps (6 rows)
```bash
python3 demo_preprocessing.py
```

Shows complete transformation:
1. Original raw data
2. After removing duplicates/NaN
3. After removing irrelevant columns
4. After label encoding
5. After feature scaling

### See Training Process (50 rows)
```bash
python3 demo_train.py
```

Shows all 4 models training and evaluation metrics.

## Quick Start Example

```python
from predict import IntrusionDetectionPredictor
import pandas as pd

# Initialize predictor
predictor = IntrusionDetectionPredictor(
    models_folder='../models',
    scaler_path='../data/processed/friday_scaler.pkl',
    encoder_path='../data/processed/friday_label_encoder.pkl'
)

# Load test data (87 features)
X_test = pd.read_csv('../data/processed/friday_X.csv').values

# Make prediction
result = predictor.predict_single_sample(X_test[0])
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## File Descriptions

### data_preprocessing.py
- Loads raw CSV files from `data/raw/`
- Removes duplicates, NaN, infinite values
- Encodes text labels to numeric
- Scales features using StandardScaler
- Saves processed data and preprocessing objects

**Key Functions:**
- `process_all_files()` - Batch process all CSV files
- `preprocess_pipeline()` - Single file preprocessing

### train.py
- Loads preprocessed data
- Splits into 80% train, 20% test (stratified)
- Trains 4 different ML models
- Saves models to pickle files

**Key Class:**
- `IntrusionDetectionTrainer` - Handles training pipeline

### evaluate.py
- Loads trained models and test data
- Calculates accuracy, precision, recall, F1-score
- Displays confusion matrix analysis
- Generates detailed classification reports

**Key Class:**
- `ModelEvaluator` - Handles evaluation pipeline

### predict.py
- Loads trained models and preprocessing objects
- Makes predictions on new data
- Supports single sample and batch predictions
- Returns predictions with confidence scores

**Key Class:**
- `IntrusionDetectionPredictor` - Handles inference pipeline

## Performance Metrics

The system evaluates models using:
- **Accuracy** - Overall correctness
- **Precision** - True positives among predicted positives
- **Recall** - True positives among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Breakdown of TP, TN, FP, FN

## Data Flow

```
Raw CSV Files (87 features)
        ↓
data_preprocessing.py
        ↓
Processed Data + Scaler + Encoder
        ↓
train.py
        ↓
Trained Models (4 algorithms)
        ↓
evaluate.py → Model Metrics
predict.py  → Predictions
```

## Web Application

The project includes a web application with both FastAPI backend and Streamlit frontend for interactive predictions and visualization.

### Streamlit Pages
- **Home** - System overview and information
- **Single Prediction** - Make individual predictions (manual entry or demo sample)
- **Batch Prediction** - Predict on multiple generated samples
- **About** - Project details and technology stack

### Quick Start - Run Web App

**Terminal 1 - Start FastAPI Backend:**
```bash
cd app
python3 -m uvicorn app:app --reload
```
API available at: http://localhost:8000 (docs at `/docs`)

**Terminal 2 - Start Streamlit Frontend:**
```bash
cd app
streamlit run streamlit_app.py
```
Web UI available at: http://localhost:8501

### FastAPI Endpoints
- `GET /health` - System health check
- `POST /predict` - Single sample prediction
- `POST /predict-batch` - Batch predictions
- `GET /models/compare` - Model information
- `GET /features/info` - Expected features info

## Dependencies

**Core Libraries:**
- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical computations
- `scikit-learn` - ML algorithms, preprocessing, metrics

**Web Framework:**
- `fastapi` - REST API backend
- `uvicorn` - ASGI web server
- `streamlit` - Interactive web UI
- `plotly` - Data visualizations

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Usage Example

```python
from predict import IntrusionDetectionPredictor
import pandas as pd

# Initialize predictor
predictor = IntrusionDetectionPredictor(
    models_folder='../models',
    scaler_path='../data/processed/friday_scaler.pkl',
    encoder_path='../data/processed/friday_label_encoder.pkl'
)

# Load test data (87 features)
X_test = pd.read_csv('../data/processed/friday_X.csv').values

# Make prediction
result = predictor.predict_single_sample(X_test[0])
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Troubleshooting

### API Backend Issues
- **"API Backend Unavailable"**: Make sure FastAPI is running on port 8000
- **Model loading errors**: Check if models are in `models/` folder
- **Missing preprocessing objects**: Check if files are in `data/processed/` folder

### Feature Dimension Errors
- Ensure your data has exactly 87 features
- CSV files should have at least 87 columns (extra columns ignored)
- Missing columns will cause prediction errors

### Environment Issues
- Install all dependencies: `pip install -r requirements.txt`
- Create and activate virtual environment before installing
- Ensure Python 3.10+ is installed

### Port Already in Use
- Change FastAPI port: `python3 -m uvicorn app:app --port 8001`
- Change Streamlit port: `streamlit run streamlit_app.py --server.port 8502`

## Performance Notes

- Single predictions: ~100-200ms
- Batch predictions: ~10-50ms per sample
- CSV upload: Depends on file size (tested with 10,000+ rows)
- All endpoints use the best-performing Random Forest model by default

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│        Streamlit Frontend (Port 8501)               │
│  - Interactive UI with 5 pages                      │
│  - Real-time predictions and visualizations         │
└────────────────┬────────────────────────────────────┘
                 │ HTTP Requests
                 ↓
┌─────────────────────────────────────────────────────┐
│       FastAPI Backend (Port 8000)                   │
│  - REST API with detailed docs at /docs            │
│  - Health checks and model info endpoints          │
└────────┬──────────────────────────────┬─────────────┘
         │                              │
         ↓                              ↓
    Load Models              Load Preprocessing Objects
    (4 algorithms)           (Scaler + Encoder)
         │                              │
         └──────────────┬───────────────┘
                        ↓
                  Make Predictions
                  Return Results
```

## Dataset Citation

This project uses the **CICIDS-2017** dataset for model training and evaluation. If you use this dataset in your research, please cite:

```bibtex
@article{cicids2017,
  title={CIC-IDS2017: A Large-Scale Labelled Dataset of Internet of Things Network Traffic},
  author={Sharafaldin, Iman and Habibi Lashkari, Arash and Ghorbani, Ali A.},
  year={2017}
}
```

**Dataset Download:**
- Official source: [CIC @ UNB](https://www.unb.ca/cic/)
- Alternative source: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

**Related Publications:**
- Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization ([ResearchGate](https://www.researchgate.net/))
- The CICIDS2017 dataset is widely used in network security and machine learning research

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Shailesh Kumar Sharma**  
Email: shaileshksharma12@gmail.com

Created as part of a Network Intrusion Detection research project.

## Citation

If you use this project in your research, please cite:

```
Network Intrusion Detection System (IDS)
Machine Learning-based classification of network traffic
Author: Shailesh Kumar Sharma
Email: shaileshksharma12@gmail.com
```