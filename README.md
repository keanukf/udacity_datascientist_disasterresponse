# Disaster Response Pipeline

This project operationalizes a full-stack machine learning pipeline for multilingual crisis communication. Building on the Figure Eight (Appen) dataset of real disaster messages, the work demonstrates end-to-end research practice: hypothesis-driven data preparation, explainable feature engineering, rigorous model benchmarking, and deployment through a lightweight Flask service suitable for rapid humanitarian response scenarios.

> **Runtime note:** Target Python 3.13. Create a 3.13.x environment before installing dependencies for consistent behavior with the hosted application.

![Project Preview](/images/preview.png)

## Research Highlights
- Curated 40+ multilabel categories that capture humanitarian intents (medical aid, infrastructure, shelter, etc.) with attention to linguistic ambiguity and class imbalance.
- Developed a reproducible ETL workflow that builds a normalized SQLite database (`DisasterResponse.db`) from raw CSV sources while preserving provenance metadata.
- Prototyped several estimators—Random Forest, AdaBoost, and tuned Gradient Boosting—using stratified cross-validation to balance F1 performance with inference latency.
- Implemented a custom tokenizer (`app/tokenizer.py`) combining lemmatization, stop-word filtering, and regex normalization to improve robustness on noisy short texts.
- Delivered an interpretable dashboard `app/templates/master.html` that surfaces per-category predictions and frequency plots for situational awareness.

## Repository Structure
- `app/` – Flask web interface, visualization logic, and custom tokenizer.
  - `templates/` – Jinja2 templates (`master.html`, `go.html`) for data exploration and classification feedback.
  - `run.py` – Entry point for the interactive dashboard.
- `data/` – Raw sources, engineered database, and ETL utilities.
  - `disaster_messages.csv`, `disaster_categories.csv` – Original Figure Eight assets.
  - `process_data.py` – Cleans, merges, and persists data to SQLite.
  - `DisasterResponse.db` – Materialized dataset used by the model.
- `models/` – Machine learning experimentation assets.
  - `train_classifier.py` – Pipeline definition, hyperparameter search, and export logic.
  - `classifier.pkl` – Serialized `sklearn` model ready for inference.
- `images/` – Visual artifacts referenced in documentation.
- `requirements.txt`, `runtime.txt`, `Procfile` – Reproducible deployment configuration.

## Experimental Methodology
1. **ETL**  
   Run the following from the project root to construct the research-grade dataset:
   ```
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
2. **Model training**  
   Train and persist the optimized multi-output classifier:
   ```
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```
   During training, the script logs per-category precision, recall, and F1 along with global micro/macro averages for transparent reporting.
3. **Evaluation**  
   Review the console metrics or extend `train_classifier.py` to emit experiment artifacts (e.g., confusion matrices, feature importances) for deeper analysis.

## Running the Interactive App
1. Move into the `app/` directory and launch the Flask server:
   ```
   cd app
   python run.py
   ```
2. Navigate to `http://0.0.0.0:3001/` (or the port displayed in the terminal).  
3. Submit free-form crisis messages; the interface annotates each humanitarian category and displays corpus-level distributions for context.

## Reproducibility Checklist
- Dependencies pinned in `requirements.txt`; tested with Python 3.13.
- Deterministic `RandomState` seeds across `process_data.py` and `train_classifier.py`.
- SQLite database + pickled model provide cold-start artifacts for immediate demoing.
- Modular tokenizer allows fast ablation studies or multilingual extensions.

## Future Research Directions
- Incorporate transformer-based encoders (e.g., multilingual BERT) to handle code-mixed inputs.
- Add active-learning hooks so analysts can relabel ambiguous samples directly from the UI.
- Integrate geospatial priors or temporal decay functions to prioritize emerging crises.