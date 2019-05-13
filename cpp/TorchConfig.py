import torch, os

if len(torch.__path__) == 0:
    print("Unable to locate pytorch package, please specify `-DCMAKE_PREFIX_PATH` in run.sh:19")
else:
    PATH = os.path.join(torch.__path__[0], "share/cmake/")
    print(PATH)
