# BiasBalance - A Dataset Bias Detection and Augmentation Tool 📊

A Streamlit-based web application for analyzing categorical data, detecting bias, performing independence testing and augmenting data.

## Features 🌟

- **Dataset Upload & Analysis**: Support for CSV file uploads with automatic categorical column detection
- **Bias Detection**: Perform Goodness-of-Fit tests to identify bias in categorical data
- **Independence Testing**: Analyze relationships between categorical variables
- **AI-powered Data Augmentation**: Enhance your dataset with intelligent data augmentation
- **Interactive Visualization**: Visual representation of test results and data distributions
- **Export Functionality**: Download augmented datasets in CSV format

## Prerequisites 📋

- Python 3.8 or higher
- pip (Python package installer)
- Docker (optional)

## Local Setup 🛠️

### Method 1: Traditional Setup

1. Clone the repository

2. Create and activate a virtual environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run src/Home.py
```

The application will be available at `http://localhost:8501`

### Method 2: Using Docker 🐳

1. Clone the repository

2. Build the Docker image
```bash
docker build -t bias-detection-tool .
```

3. Run the container
```bash
docker compose up
```

The application will be available at `http://localhost:8501`

## Environment Variables 🔐

Create a `.env` file in the root directory with the following variables (if needed):
```
OPENAI_API_KEY=your_api_key_here
```

## Usage Guide 📖

1. Launch the application
2. Upload your CSV dataset using the file uploader
3. Categorical Columns will be already selected for you. (make changes if needed)
4. Use the sidebar to navigate between different testing options
5. Perform Goodness-of-Fit tests to detect bias and add for augmentation if bias is detected
6. Analyze relationships between categorical variables using the Independence Testing feature
7. Use the Data Augmentation feature to enhance your dataset
8. View results and download augmented datasets if needed