# Summary
In this project, I use ETL and ML pipelines to clean, process and classify text messages sent during disaster events into multiple categories pertaining to disaster response.

# File Description
- disaster_categories.csv - Contains data on disaster response categories and how text messages by ID are classified for them
- disaster_messages.csv - Contains list of texts sent during disaster events listed by ID
- process_data.py - Extracts text and classification data from CSV files, then cleans and loads data to SQLite database
- train_classifier.py - Set up ML pipeline that runs count vectorizer, TF-IDF and random forest classifier on cleaned text data loaded from SQLite database to predict disaster category classification

# Instructions
1. To process the data, open the Terminal and run the command "python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db".
2. To create and run the ML pipeline, run the command "python train_classifier.py DisasterResponse.db classifier.pkl".
