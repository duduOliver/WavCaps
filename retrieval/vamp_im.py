import vampnet
import os
from pathlib import Path

print(Path(__file__).parent.parent.parent)
print("#########")
import sys
# os.path.split(os.getcwd())[0]
# sys.path.append("/gpfs/home/dhuang/thesis/WavCaps/retrieval/models")
# sys.path.append("/gpfs/home/dhuang/thesis/WavCaps/retrieval/tools")
for nb_dir in sys.path:
    print(nb_dir)
    # sys.path.append(nb_dir)

print("os.getcwd(): ", os.getcwd())
# os.path.split(os.getcwd())[0]