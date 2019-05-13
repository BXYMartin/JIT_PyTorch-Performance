import threading, pynvml
import torch
from torchvision import models
import time
import sys

resultArray=[]
testNum = 50

class Tester:
    def get_time(self):
        return time.time() * 1000

    def _test_model(self, name, input_shape):
        traced_model = torch.jit.load(name).eval()
        traced_model.cuda()
        traced_model(torch.rand(input_shape).cuda())

        torch.cuda.synchronize()
        trace_start_time=self.get_time()
        for i in range(testNum):
            traced_model(torch.empty(input_shape).cuda())
        torch.cuda.synchronize()
        trace_end_time=self.get_time()


        del traced_model

        print("Name:{0:>12} Batch:{1:>3d} Eval:{2:>12.3f}ms Trace:{3:>12.3f}ms Eval Cuda Size:{4:>10d}KB Traced Cuda Size:{5:>10d}KB".format(
            name,
            input_shape[0],
            0 / testNum,
            (trace_end_time - trace_start_time) / testNum,
            0,
            0,
        ))

        sys.stdout.flush()
        resultArray.append("Name:{0:>12} Batch:{1:>3d} Eval:{2:>12.3f}ms Trace:{3:>12.3f}ms Eval Cuda Size:{4:>10d}KB Traced Cuda Size:{5:>10d}KB".format(
            name,
            input_shape[0],
            0 / testNum,
            (trace_end_time - trace_start_time) / testNum,
            0,
            0,
        ))


def get_available_models():
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0]]

if __name__ == '__main__':
    t = Tester()
    #for model_name in ['alexnet']:
    for model_name in get_available_models():
        for batch_size in [1, 2, 4, 8, 16]:
            input_shape = (batch_size, 3, 224, 224)
            if model_name in ['inception_v3']:
                input_shape = (batch_size, 3, 299, 299)
            t._test_model(model_name, input_shape)
    with open("jit.txt","w+") as result:
        print("[======================================================]")
        for line in resultArray:
            print(line)
            result.write(line + "\n")
        print("[======================================================]")
