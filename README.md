# Network Intrusion Detection System (IDS)

A machine learning-based Network Intrusion Detection System that classifies network traffic as benign or malicious attacks. This project includes a complete ML pipeline with FastAPI backend and Streamlit web interface for interactive predictions.

## Project Overview

This project implements:
- **Data Preprocessing**: Clean and normalize raw network traffic data from CSV files
- **Model Training**: Train Random Forest classifier for multi-class attack detection
- **Model Evaluation**: Comprehensive performance analysis with multiple metrics
- **Inference Pipeline**: Make predictions on new network traffic
- **Web API**: FastAPI REST endpoints for programmatic access
- **Web UI**: Streamlit interface for interactive predictions and visualization

## Dataset

### Network Traffic Data
This project uses network traffic datasets with daily captures, including both benign and attack traffic.

**Dataset Details:**
- **Collection Period**: Multiple days (Monday through Friday)
- **Total Samples**: Thousands of traffic flows per day
- **Features**: 82 network characteristics (after preprocessing)
- **Attack Types**: 5 types of attacks plus benign traffic

### Traffic Classification

The system classifies network traffic into 6 categories:
- **Benign**: Normal, non-malicious traffic
- **DoS**: Denial of Service attacks
- **DDoS**: Distributed Denial of Service attacks
- **Probe**: Reconnaissance/probing attacks
- **R2L**: Remote to Local attacks
- **U2R**: User to Root escalation attacks

Each sample contains 82 network features extracted from traffic flows after removing non-predictive columns like Flow ID, Source/Destination IP, and Timestamp.

## Machine Learning Model

**Random Forest Classifier**
- **Estimators**: 200 decision trees
- **Max Depth**: 25
- **Feature Strategy**: sqrt (reduce tree correlation)
- **Class Weights**: Balanced (handle class imbalance)
- **Optimization**: Stratified train/test split (80/20)

## Project Structure

```
multiclass-attack-classifier/
├── data/
│   ├── raw/                      # Original CSV files
│   │   ├── monday.csv
│   │   ├── friday.csv
│   │   ├── tuesday.csv
│   │   ├── wednesday.csv
│   │   ├── thursday.csv
│   │   └── *_plus.csv            # Extended datasets
│   └── processed/                # Clean, normalized data + preprocessing objects
│       ├── *_X.csv               # Features
│       ├── *_y.csv               # Labels
│       └── *_scaler.pkl          # StandardScaler objects
├── models/                       # Trained ML models (joblib format)
│   ├── random_forest_model.joblib
│   └── feature_scaler.joblib
├── src/
│   ├── data_preprocessing.py     # Clean, normalize, and prepare data
│   ├── train.py                  # Train Random Forest model
│   ├── evaluate.py               # Evaluate model performance
│   └── predict.py                # Make predictions on new data
├── app/
│   ├── app.py                    # FastAPI backend with REST endpoints
│   ├── streamlit_app.py          # Streamlit web interface
│   └── README.md                 # Web application documentation
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis
├── reports/
│   ├── evaluation_report.txt     # Model evaluation metrics
│   └── predictions_sample.csv    # Sample predictions
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multiclass-attack-classifier.git
cd multiclass-attack-classifier
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
- `{filename}_X.csv` - Features (82 columns)
- `{filename}_y.csv` - Labels (attack types)
- `{filename}_scaler.pkl` - StandardScaler object

### 2. Train Model
Train Random Forest model on preprocessed data:
```bash
python3 train.py
```

Models saved to `models/`:
- `random_forest_model.joblib` - Trained classifier
- `feature_scaler.joblib` - Feature scaler

### 3. Evaluate Model
Evaluate trained model and view performance metrics:
```bash
python3 evaluate.py
```

Displays:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Detailed classification report

### 4. Make Predictions
Predict on new network traffic data:
```bash
python3 predict.py
```

## Web Application

### Run FastAPI Backend
```bash
cd app
python3 -m uvicorn app:app --reload
```

API will be available at: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Run Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

Web UI will be available at: `http://localhost:8501`

## API Endpoints

### Single Prediction
**POST** `/predict`
```json
{
  "features": [1.0, 2.0, 3.0, ..., 82_values]
}
```
Returns: `{"prediction": "DoS", "confidence": 0.95, "model_used": "Random Forest"}`

### Batch Prediction
**POST** `/predict-batch`
```json
{
  "samples": [[82 features], [82 features], ...]
}
```
Returns: Predictions with statistics (benign count, attack count, etc.)

### CSV Upload
**POST** `/predict-csv`
Upload a CSV file with 82 columns (network features)

### Health Check
**GET** `/health`
Returns: Status of model and scaler

### Model Information
**GET** `/models/compare`
Returns: Details about loaded models

### Feature Information
**GET** `/features/info`
Returns: Expected features and preprocessing details

## Module Descriptions

### data_preprocessing.py
Handles complete data preprocessing pipeline:
- Load raw CSV files
- Remove duplicates, NaN, and infinite values
- Drop irrelevant columns (IPs, timestamps, etc.)
- Encode categorical labels
- Scale features using StandardScaler
- Save processed data and preprocessing objects

**Key Functions:**
- `load_data()` - Load CSV into DataFrame
- `clean_data()` - Remove bad records
- `remove_irrelevant_columns()` - Drop non-predictive columns
- `encode_labels()` - Convert text labels to numeric
- `scale_features()` - Normalize features
- `preprocess_pipeline()` - Orchestrate all steps
- `process_all_files()` - Batch process multiple files

### train.py
Training pipeline for Random Forest model:
- Load preprocessed data
- Split into train/test sets (80/20 stratified)
- Train Random Forest with optimized hyperparameters
- Evaluate on test set
- Save model and scaler

**Key Functions:**
- `load_preprocessed_data()` - Load features and labels
- `train_random_forest()` - Train and evaluate model
- `save_model()` - Save model and scaler to disk

### evaluate.py
Model evaluation and analysis:
- Load trained model and test data
- Calculate multiple metrics (accuracy, precision, recall, F1)
- Generate confusion matrix
- Create detailed classification reports
- Save evaluation report

**Key Functions:**
- `load_model_and_data()` - Load model and test data
- `evaluate_model()` - Calculate performance metrics
- `generate_report()` - Create detailed evaluation report
- `save_evaluation_report()` - Save results to file

### predict.py
Inference pipeline for making predictions:
- Load trained model and scaler
- Preprocess new data (convert DataFrame to numpy, scale)
- Make predictions with confidence scores
- Support single and batch predictions
- Handle different input formats

**Key Functions:**
- `load_model_and_scaler()` - Load trained objects
- `preprocess_input()` - Scale input data
- `predict_single()` - Predict single sample
- `predict_batch()` - Predict multiple samples
- `predict_with_confidence()` - Get predictions with probabilities
- `predict_traffic_flow()` - Predict network flow
- `batch_prediction_from_csv()` - Batch predict from CSV

## Performance Metrics

The system evaluates model performance using:
- **Accuracy** - Overall correctness across all classes
- **Precision** - True positives among predicted positives (per class)
- **Recall** - True positives among actual positives (per class)
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Breakdown of TP, TN, FP, FN for each class
- **Support** - Number of samples in each class

## Data Pipeline

```
Raw CSV Files (with metadata columns)
        ↓
data_preprocessing.py
        ↓
Clean Data (remove duplicates, NaN, infinite values)
        ↓
Remove Irrelevant Columns (IPs, timestamps, identifiers)
        ↓
Encode Labels (text → numeric)
        ↓
Scale Features (StandardScaler: mean=0, std=1)
        ↓
Processed Data (82 features) + Scaler Object
        ↓
train.py
        ↓
Trained Random Forest Model
        ↓
evaluate.py → Model Metrics & Reports
predict.py  → Make Predictions on New Data
app.py      → REST API Endpoints
streamlit_app.py → Interactive Web UI
```

## Key Features

✅ **Automated Data Preprocessing**: Complete cleaning and normalization pipeline  
✅ **Multi-class Classification**: Detect 6 different traffic types  
✅ **High Performance**: Random Forest with 200 trees  
✅ **Confidence Scores**: Get prediction probability for uncertainty assessment  
✅ **Batch Processing**: Predict on multiple samples efficiently  
✅ **REST API**: FastAPI for programmatic access  
✅ **Web Interface**: Streamlit UI for interactive predictions  
✅ **Detailed Reports**: Comprehensive evaluation metrics and visualizations  

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn (data and ML)
- joblib (model serialization)
- fastapi, uvicorn (REST API)
- streamlit, plotly (web interface)
- requests (API client)

See `requirements.txt` for exact versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Network Intrusion Detection System Project  
Built with Python, scikit-learn, FastAPI, and Streamlit

## Acknowledgments

- Dataset collection and preprocessing techniques inspired by industry best practices
- Random Forest hyperparameters optimized for network intrusion detection
- Web interface built with modern Python web frameworks

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