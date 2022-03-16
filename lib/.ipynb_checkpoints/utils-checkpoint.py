import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def change_portuguese_to_english_cols(df):

    df['max_valor_boleto'] =  df.max_valor_boleto.str.replace(',','.').astype(np.float32)

    df['avg_valor_boleto'] =  df.avg_valor_boleto.str.replace(',','.').astype(np.float32)

    df['total_valor_boleto'] =  df.total_valor_boleto.str.replace(',','.').astype(np.float32)

    df['valor_boleto_stdv'] =  df.valor_boleto_stdv.str.replace(',','.').astype(np.float32)
    return df


def plot_confusion_matrix(y_test, p, ax=None):
    cm = confusion_matrix(y_test, p > 0.5)

    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    target_names = ['non-fraud','fraud']
    sns.heatmap(cmn, annot=True, fmt='.4f', xticklabels=target_names, yticklabels=target_names,ax=None)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
def print_results(y, prob):
    print("Classification report:")
    print(classification_report(y,prob > 0.5,digits=4))
    print("ROC AUC Score:")
    print("\t",round(roc_auc_score(y, prob),4))
    print("Average precision-recall score:")
    print("\t",round(average_precision_score(y, prob),4))
    
def print_results_model(y, model, X):
    y_pred = model.predict_proba(X)[:,1]
    
    print("Classification report:")
    print(classification_report(y, y_pred > 0.5,digits=4))
    print("ROC AUC Score:")
    print("\t",round(roc_auc_score(y, y_pred),4))
    print("Average precision-recall score:")
    print("\t",round(average_precision_score(y, y_pred > 0.5),4))    
    
def plot_confusion_matrix_model(y_test, model, X):
    y_pred = model.predict_proba(X)[:,1]
    cm = confusion_matrix(y_test, y_pred > 0.5)

    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    target_names = ['non-fraud','fraud']
    sns.heatmap(cmn, annot=True, fmt='.4f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
        