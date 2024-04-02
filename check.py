import torch
import sys

print(torch.__version__)
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available. Continuing execution...")
    # If CUDA is not available, you can decide to exit the script or continue
    # sys.exit()
