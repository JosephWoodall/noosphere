
import json
import numpy as np
from pathlib import Path

def analyze_class_balance(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print(f"{'Dataset':<20} | {'Mode':<15} | {'Class Distribution (Preds)'}")
    print("-" * 70)
    
    for ds_name, ds_data in data.get('datasets', {}).items():
        for mode in ['within_subject', 'loso']:
            if mode not in ds_data: continue
            
            # Look at confusion matrix to see if model is collapsing
            cm = np.array(ds_data[mode]['s4']['confusion_matrix'])
            pred_sums = cm.sum(axis=0)
            total = pred_sums.sum()
            dist = [f"{p/total:.1%}" for p in pred_sums]
            
            acc = ds_data[mode]['s4']['accuracy']
            print(f"{ds_name:<20} | {mode:<15} | {dist} (Acc: {acc:.1%})")

if __name__ == "__main__":
    analyze_class_balance('real_eeg_results.json')
