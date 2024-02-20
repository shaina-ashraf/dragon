import torch
import sys


print(torch.__version__)
# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is available. ...")
    print(torch.version.cuda)
    #sys.exit()
else:
    # If CUDA is available, the rest of your code will run below this point
    print("CUDA is not available. Continuing execution...")
    # Your remaining code here
