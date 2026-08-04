"""
plot_results.py: generates 4 result figures for the IAPA report.

figures saved to diagrams/:
  fig1_classification_report.png   precision / recall / F1 per category (AI bot)
  fig2_confusion_matrix.png        where the model gets confused (billing vs refund)
  fig3_rpa_validation.png          115 pass vs 5 fail, broken by rule
  fig4_confidence_scores.png       confidence for every routed ticket + the 0.55 threshold

run:  python3 plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import joblib
import html
from datetime import datetime

from Config import (
    TICKETS_PATH, CUSTOMERS_PATH, EXCEPTIONS_PATH, ROUTED_TICKETS_PATH,
    TICKET_TEXT, CATEGORY, CONF_THRESHOLD, VECTORIZER_PATH, MODEL_PATH,
    VALID_PREFIXES, MSISDN_LENGTH, VALID_SEGMENTS, VALID_STATUSES,
    REQUIRED_CUSTOMER_COLUMNS,
)

DIAGRAMS_DIR = 'diagrams'
os.makedirs(DIAGRAMS_DIR, exist_ok=True)

TEAL   = '#2E6E6E'
ORANGE = '#E07B39'
GREY   = '#8A94A6'
RED    = '#C0392B'
GREEN  = '#27AE60'

# helpers

def clean_text(text):
    return html.unescape(str(text).lower())

def retrain_and_get_metrics():
    df = pd.read_csv(TICKETS_PATH)
    df[TICKET_TEXT] = df[TICKET_TEXT].fillna('').apply(clean_text)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[CATEGORY]
    )
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(df_train[TICKET_TEXT])
    X_test  = vec.transform(df_test[TICKET_TEXT])
    nb = MultinomialNB()
    nb.fit(X_train, df_train[CATEGORY])
    y_pred = nb.predict(X_test)
    y_true = df_test[CATEGORY].values
    return y_true, y_pred, nb.classes_

# FIG 1, precision / recall / F1 per category

def fig1_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, output_dict=True)
    labels   = [c for c in classes]
    short    = {'SIM_ACTIVATION':'SIM', 'NETWORK':'NET',
                'PAYMENT_BILLING':'P_BILL', 'REFUNDS':'REFUND', 'GENERAL':'GEN'}
    x_labels = [short.get(l, l) for l in labels]
    precision = [report[l]['precision'] for l in labels]
    recall    = [report[l]['recall']    for l in labels]
    f1        = [report[l]['f1-score']  for l in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(False)
    ax.bar(x - w, precision, w, label='Precision', color=TEAL)
    ax.bar(x,     recall,    w, label='Recall',    color=ORANGE)
    ax.bar(x + w, f1,        w, label='F1-Score',  color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('AI Bot (triage_bot.py): Classification Report per Category')
    ax.legend()
    ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - w, p + 0.02, f'{p:.2f}', ha='center', fontsize=8)
        ax.text(i,     r + 0.02, f'{r:.2f}', ha='center', fontsize=8)
        ax.text(i + w, f + 0.02, f'{f:.2f}', ha='center', fontsize=8)
    plt.tight_layout()
    path = f'{DIAGRAMS_DIR}/fig1_classification_report.png'
    plt.savefig(path, dpi=130)
    plt.close()
    print(f'[saved] {path}')

# FIG 2, confusion matrix

def fig2_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    short = {'SIM_ACTIVATION':'SIM', 'NETWORK':'NET',
             'PAYMENT_BILLING':'P_BILL', 'REFUNDS':'REFUND', 'GENERAL':'GEN'}
    labels = [short.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(False)
    im = ax.imshow(cm, cmap='YlOrRd')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual',    fontsize=11)
    ax.set_title('AI Bot: Confusion Matrix\n(billing vs refund is the only blur)')
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = f'{DIAGRAMS_DIR}/fig2_confusion_matrix.png'
    plt.savefig(path, dpi=130)
    plt.close()
    print(f'[saved] {path}')

# FIG 3, RPA validation pass / fail

def fig3_rpa_validation():
    df_exc = pd.read_csv(EXCEPTIONS_PATH)
    df_all = pd.read_csv(CUSTOMERS_PATH, dtype={'MSISDN': str})
    total  = len(df_all)
    failed = len(df_exc)
    passed = total - failed

    # break down which rule caught each exception
    rule_counts = {
        'Missing field':     0,
        'Bad MSISDN digits': 0,
        'Bad MSISDN prefix': 0,
        'Unknown segment':   0,
        'Unknown status':    0,
        'Bad date':          0,
    }
    for err in df_exc['validation_errors'].fillna(''):
        if 'is empty'           in err: rule_counts['Missing field']     += 1
        if 'not 11 digits'      in err: rule_counts['Bad MSISDN digits'] += 1
        if 'prefix'             in err: rule_counts['Bad MSISDN prefix'] += 1
        if 'unknown segment'    in err.lower(): rule_counts['Unknown segment']   += 1
        if 'unknown status'     in err.lower(): rule_counts['Unknown status']    += 1
        if 'bad'                in err.lower() and 'date' in err.lower(): rule_counts['Bad date'] += 1

    rule_counts = {k: v for k, v in rule_counts.items() if v > 0}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # left: pass vs fail donut
    ax = axes[0]
    sizes  = [passed, failed]
    colors = [GREEN, RED]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[f'Clean\n{passed}', f'Exception\n{failed}'],
        colors=colors, autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(width=0.55)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
    ax.set_title(f'RPA Bot: {total} Records Validated')

    # right: which rule caught each exception
    ax2 = axes[1]
    rules  = list(rule_counts.keys())
    counts = list(rule_counts.values())
    ax2.grid(False)
    bars = ax2.barh(rules, counts, color=RED, alpha=0.8)
    ax2.set_xlabel('Records caught')
    ax2.set_title('Exceptions by Validation Rule')
    ax2.set_xlim(0, max(counts) + 1)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    path = f'{DIAGRAMS_DIR}/fig3_rpa_validation.png'
    plt.savefig(path, dpi=130)
    plt.close()
    print(f'[saved] {path}')

# FIG 4, confidence scores

def fig4_confidence():
    df = pd.read_csv(ROUTED_TICKETS_PATH)
    if 'confidence' not in df.columns:
        print('[skip] routed_tickets.csv has no confidence column, run triage_bot.py route first')
        return

    confidences = df['confidence'].values
    flagged     = df['needs_human_review'].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(False)
    colors = [RED if f else TEAL for f in flagged]
    x = range(len(confidences))
    ax.bar(x, confidences, color=colors, alpha=0.85)
    ax.axhline(CONF_THRESHOLD, color='black', linestyle='--', linewidth=1.5,
               label=f'Confidence threshold ({CONF_THRESHOLD})')
    ax.set_xlabel('Ticket index')
    ax.set_ylabel('Confidence score')
    ax.set_title('AI Bot: Confidence Score per Routed Ticket')
    ax.set_ylim(0, 1.1)
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color=TEAL, label='Auto-routed'),
        Patch(color=RED,  label='Flagged for human review'),
        plt.Line2D([0], [0], color='black', linestyle='--', label=f'Threshold ({CONF_THRESHOLD})')
    ]
    ax.legend(handles=legend_els)
    plt.tight_layout()
    path = f'{DIAGRAMS_DIR}/fig4_confidence_scores.png'
    plt.savefig(path, dpi=130)
    plt.close()
    print(f'[saved] {path}')

# main

def main():
    print('retraining AI bot to capture metrics...')
    y_true, y_pred, classes = retrain_and_get_metrics()
    print(f'classes: {classes}')

    print('\ngenerating figures...')
    fig1_classification_report(y_true, y_pred, classes)
    fig2_confusion_matrix(y_true, y_pred, classes)
    fig3_rpa_validation()
    fig4_confidence()
    print('\ndone, 4 figures saved to diagrams/')

if __name__ == '__main__':
    main()
