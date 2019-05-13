import threading, pynvml
import torch
from torchvision import models
import time
import os

class Tester:
    def _test_model(self, name, input_shape):
        model = models.__dict__[name]()
        model.eval()
        model.cuda()

        traced_model = torch.jit.trace(model, torch.empty(input_shape).cuda()).eval()
        traced_model.save(os.path.join("./modules", name))

        print("[-] Generated: {0:>12}".format(name))


def get_available_models():
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0]]

if __name__ == '__main__':
    if not os.path.exists("modules"):
        os.mkdir("modules")
    t = Tester()
    for model_name in get_available_models():
        if model_name in os.listdir("modules"):
            continue
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        t._test_model(model_name, input_shape)
