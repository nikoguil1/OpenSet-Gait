import pickle
import numpy as np

pkl_path = '/home/nico/gait/datasets/Gait3D/Gait3D-pkls/Gait3D-smpls-pkl/0125/camid14_videoid3/seq0/seq0.pkl'

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Data type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                if isinstance(data[0], np.ndarray):
                    print(f"First element shape: {data[0].shape}")
        else:
            print(f"Content: {data}")
except Exception as e:
    print(f"Error loading pickle: {e}")
