import glob
import pickle
import numpy as np

for f in sorted(glob.glob("/home/nico/gait/OpenGait/roc_data/embeddings_Gait3D_DeepGaitV2_*.pkl")):
    try:
        with open(f, "rb") as file:
            data = pickle.load(file)
            print(f"--- {f} ---")
            # print(f"  Keys: {list(data.keys())}")
            
            if 'EER' in data:
                print(f"  EER = {data['EER']*100:.2f}%")
            if 'FPR' in data and 'TPR' in data:
                fpr = data['FPR']
                tpr = data['TPR']
                tar_1 = np.interp(0.01, fpr, tpr)
                tar_01 = np.interp(0.001, fpr, tpr)
                print(f"  TAR@1FAR = {tar_1*100:.2f}%")
                print(f"  TAR@0.1FAR = {tar_01*100:.2f}%")
            if 'Rank-1' in data:
                print(f"  Rank-1 = {data['Rank-1']}")
            if 'mAP' in data:
                print(f"  mAP = {data['mAP']}")
                
    except Exception as e:
        print(f"Error loading {f}: {e}")
