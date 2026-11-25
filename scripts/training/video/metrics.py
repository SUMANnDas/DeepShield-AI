from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix

def binary_metrics(y_true,y_prob):
    import numpy as np
    y=np.asarray(y_true); p=np.asarray(y_prob); pred=(p>=.5).astype(int); pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(pr),'recall':float(re),'f1':float(f1),'roc_auc':float(roc_auc_score(y,p)) if len(set(y.tolist()))==2 else None,'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
