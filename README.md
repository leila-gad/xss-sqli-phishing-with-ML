# IDS using Machine Learning
Web attack detection is critical for securing systems. Traditional rule-based IDSs are often limited against new threats.
This project implements an Intrusion Detection System (IDS) using Machine Learning to detect and classify common web attacks: XSS, SQL Injection, and Phishing.
The project leverages Machine Learning to:
        -->Analyze textual payloads (URLs, queries, scripts)
        -->Classify attacks and benign traffic
        -->Provide predictions via an API and interactive frontend

Here is the full ML project lifecycle: data preparation, model training, evaluation, API deployment, frontend interface, and Docker containerization.

ML Pipeline
        /Data Preparation: 
        Unified datasets for XSS, SQLi, Phishing
        Numeric label mapping: 0 = XSS, 1 = SQLi, 2 = Phishing
        Generates train.csv and test.csv

        /Feature Extraction: 
        TF-IDF on n-grams (1–3)
        Converts payloads into numeric vectors

        /Model Training:
        XGBoost for multi-class classification
        Stratified train/test split
        Saves model and vectorizer

        /Evaluation
        Accuracy, Precision, Recall, F1-score
        Confusion matrix & error analysis

Frontend Features:     
        Input payload to test
        Request history
        Simplified dashboard for quick visualization

Results & Discussion:
        Accurate detection of XSS, SQLi, and Phishing attacks
        Limitations: dependent on dataset quality & diversity
        Future improvements: add new attacks, continuous learning, SIEM integration, cloud deployment


Installation and running:
        /Clone the repository:
        git clone https://github.com/leila-gad/xss-sqli-phishing-with-ML.git
        cd xss-sqli-phishing-with-ML

        /Create a virtual environment:
        python -m venv venv

        /activate the virtual environment
        # Windows PowerShell
        venv\Scripts\Activate.ps1
        # Linux / macOS
        source venv/bin/activate

        /Install dependencies
        pip install -r requirements.txt

        /Run the API
        cd app
        uvicorn main:app --reload

        /Open the Frontend
        Open app/index.html in your browser or run a local server.
        it can open at something like http://127.0.0.1:5500
        

        You can now test the dashboard:) type a payload → see prediction and logs








