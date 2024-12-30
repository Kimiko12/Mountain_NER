import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    """
    Calculation of metrics (precision, recall, f1, accuracy) for the NER task.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy() 
    labels = labels 

    assert predictions.shape == labels.shape, "Размерности predictions и labels должны совпадать"
    true_labels = []
    pred_labels = []

    for i in range(labels.shape[0]):  
        for j in range(labels.shape[1]): 
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                pred_labels.append(predictions[i][j])

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro'
    )
    acc = accuracy_score(true_labels, pred_labels)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

            


