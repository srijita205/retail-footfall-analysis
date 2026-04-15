import scipy.io
import pandas as pd

# Load frame names (converts frames to time reference)
with open(r"Results\frame_names.txt", "r") as f:
    frame_names = f.readlines()

print(f"Total frames: {len(frame_names)}")
print("Sample frames:", frame_names[:3])

# Load one detected action file (action 1 = Reach to Shelf)
action1 = scipy.io.loadmat(r"Results\DetectedActions\1.mat")
print("\nKeys in action file:", action1.keys())

