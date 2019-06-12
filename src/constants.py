"""Constants.

Contains constants that may be used by other classes in one class.
"""
import numpy as np
import torch
from sklearn.preprocessing import normalize



DIM_WIDTH = 380  # Ideally, this would be 640 if it could fit in the GPU memory
DIMENSIONS = (DIM_WIDTH, int(float(DIM_WIDTH) * .75))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Create a normalized index array based on the dimensions
index_array = np.indices((DIMENSIONS[1], DIMENSIONS[0]))
NORMALIZED_INDICES = []
# Normalize each axis independently
for axis in index_array:
    NORMALIZED_INDICES.append(normalize(axis, axis=0, norm="max"))
# Combine the axes back into a single np array
NORMALIZED_INDICES = torch.from_numpy(np.array(NORMALIZED_INDICES))\
    .to(dtype=torch.float, device=DEVICE)
