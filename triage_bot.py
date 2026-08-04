"""
triage_bot.py: the AI / agentic bot (Solution 2 of the IAPA CA).

Reads free-text customer-care tickets, predicts which of 5 categories each one
belongs to (TF-IDF -> Multinomial Naive Bayes), routes it to the right backend
team, and flags low-confidence tickets for a human (human-in-the-loop).

Run:  python3 triage_bot.py train                       (fit + evaluate + save artifacts)
      python3 triage_bot.py route                       (route data/new_tickets.csv)
      python3 triage_bot.py route --input x.csv --output y.csv
"""

import html
import argparse
from datetime import datetime

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from Config import (
    TICKETS_PATH,
    NEW_TICKETS_PATH,
    ROUTED_TICKETS_PATH,
    TICKET_TEXT,
    CATEGORY,
    REQUIRED_TICKET_COLUMNS,
    CONF_THRESHOLD,
    VECTORIZER_PATH,
    MODEL_PATH,
    ROUTING,
)

#shared helpers

def clean_text(text: str):
    #lowercase everything so "SIM" and "sim" look the same to the model
    text = text.lower()

    #turn &amp; and friends back into normal characters
    text = html.unescape(text)

    return text


def load_data():
    #read the tickets csv into a dataframe
    df = pd.read_csv(TICKETS_PATH)

    #fill any missing text with an empty string, just in case
    df[TICKET_TEXT] = df[TICKET_TEXT].fillna('')

    #validation, make sure the columns we need are actually there
    for col in REQUIRED_TICKET_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the tickets csv.")
    return df


def fit_vectorizer(texts):
    #create the TF-IDF object (turns words into numbers)
    vectorizer = TfidfVectorizer()

    #learn the word list from the training text and convert it to numbers
    X_features = vectorizer.fit_transform(texts)

    #save the learned vocabulary to disk so route reuses the exact same one
    joblib.dump(vectorizer, VECTORIZER_PATH)

    #hand back the numbers we just made
    return X_features


#TRAIN side

def prepare_and_split_data():
    df = load_data()

    #clean the text column with the same tool the route side will use
    df[TICKET_TEXT] = df[TICKET_TEXT].apply(clean_text)

    #split 80/20, stratified so each category keeps its fair share in both halves
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[CATEGORY]
    )
    return df_train, df_test


def vectorize_data(df_train, df_test):
    #1. fit TF-IDF on train only, save the artifact, get train features
    X_train = fit_vectorizer(df_train[TICKET_TEXT])

    #2. reload the artifact to prove the save/load round-trip works
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)

    #3. transform test using the learned vocabulary only, zero leakage
    X_test = loaded_vectorizer.transform(df_test[TICKET_TEXT])

    return X_train, X_test


def train_model(X_train, df_train):
    print("training multinomial naive bayes...")

    #NB is the actual classifier; my first probabilistic/generative model
    nb = MultinomialNB()

    #fit it on the training numbers and their categories
    nb.fit(X_train, df_train[CATEGORY])

    #save the trained model to disk
    joblib.dump(nb, MODEL_PATH)
    print(f"✓ fitted and saved naive bayes to {MODEL_PATH}")

    return nb


def evaluate_model(nb, X_test, df_test):
    print("\nevaluating on the held-out test set...")

    #ask the model to predict categories for the test numbers
    y_pred = nb.predict(X_test)
    y_true = df_test[CATEGORY]

    #how many did it get right overall
    acc = accuracy_score(y_true, y_pred)
    print(f"test accuracy: {acc:.3f}")

    #per-class precision / recall / f1
    print(classification_report(y_true, y_pred))

    return acc


#ROUTE side

def prepare_and_vectorize(input_path):
    print(f"loading new tickets from {input_path}...")

    #load the new tickets
    df_new = pd.read_csv(input_path)

    #handle missing text just in case
    df_new[TICKET_TEXT] = df_new[TICKET_TEXT].fillna('')

    #clean the text with the exact same tool the train side used
    df_new[TICKET_TEXT] = df_new[TICKET_TEXT].apply(clean_text)

    #load the fitted vectorizer artifact, no fit() here
    print("loading vectorizer artifact...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    #turn the new text into numbers using the learned vocabulary
    print("vectorizing new text...")
    X_new = vectorizer.transform(df_new[TICKET_TEXT])

    return df_new, X_new


def predict_and_route(df_new, X_new):
    print("\npredicting categories and routing...")

    #load the trained naive bayes model back from disk
    nb = joblib.load(MODEL_PATH)

    #predict a category for each new ticket
    df_new['predicted_category'] = nb.predict(X_new)

    #confidence = the probability of the top pick, for example "i'm 0.68 sure this is billing"
    df_new['confidence'] = nb.predict_proba(X_new).max(axis=1)

    #map the predicted category to the backend team
    df_new['routed_team'] = df_new['predicted_category'].map(ROUTING)

    #below my 0.55 line, the bot is not sure enough; let a human check
    df_new['needs_human_review'] = df_new['confidence'] < CONF_THRESHOLD
    flagged = df_new['needs_human_review'].sum()
    print(f"routing complete: {flagged} tickets flagged for human review.")

    return df_new


def finalize_and_save(df_new, output_path):
    print("\nfinalizing output and saving to csv...")

    #stamp which model version and exactly when
    df_new['model_version'] = 'nb_v1'
    df_new['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #the columns we want in the final file (only keep the ones that exist)
    final_columns = [c for c in ['ticket_id', TICKET_TEXT, 'predicted_category',
                     'confidence', 'routed_team', 'needs_human_review',
                     'model_version', 'prediction_timestamp'] if c in df_new.columns]
    df_final = df_new[final_columns]

    #write it out
    df_final.to_csv(output_path, index=False)
    print(f"saved routed tickets to: {output_path}")

    return df_final


#MAIN EXECUTION
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer-care triage bot: train the model, then route new tickets.")
    parser.add_argument('mode', choices=['train', 'route'], help="train = fit + evaluate + save | route = classify + route new tickets")
    parser.add_argument('--input', type=str, default=NEW_TICKETS_PATH, help="Path to input csv of new tickets (route mode)")
    parser.add_argument('--output', type=str, default=ROUTED_TICKETS_PATH, help="Path to output csv (route mode)")
    args = parser.parse_args()

    if args.mode == 'train':
        #1. get the cleanly split DataFrames
        df_train, df_test = prepare_and_split_data()

        #2. vectorize them safely
        X_train, X_test = vectorize_data(df_train, df_test)
        print(f"train features shape: {X_train.shape}")
        print(f"test features shape: {X_test.shape}")

        #3. train and save the model
        nb = train_model(X_train, df_train)

        #4. see how well it does
        evaluate_model(nb, X_test, df_test)

    else:
        #1. load + vectorize the new tickets
        df_new, X_new = prepare_and_vectorize(args.input)
        print(f"loaded {len(df_new)} new tickets.")

        #2. predict a category and route each one
        df_new = predict_and_route(df_new, X_new)

        #3. save the routed output
        finalize_and_save(df_new, args.output)

    print("\ndone.")
