import threading, pynvml
import torch
from torchvision import models
import time

resultArray=[]
testNum = 5

class Watcher:
    id = torch.cuda.current_device()
    peak_monitoring = False
    nvml_peak = 0
    nvml_start = 0

    def gpu_mem_used(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used/2**20)

    def gpu_mem_used_no_cache(self):
        torch.cuda.empty_cache()
        return self.gpu_mem_used()

    def peak_monitor_start(self):
        self.nvml_start = self.gpu_mem_used_no_cache()
        print(self.nvml_start)
        self.peak_monitoring = True

        # this thread samples RAM usage as long as the current epoch of the fit loop is running
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def peak_monitor_stop(self):
        self.peak_monitoring = False
        return self.nvml_peak - self.nvml_start

    def peak_monitor_func(self):
        self.nvml_peak = 0
        i=0
        while True:
            if i%10 == 0:
                print(self.gpu_mem_used())
            i+=1
            self.nvml_peak = max(self.gpu_mem_used(), self.nvml_peak)
            if not self.peak_monitoring: break
            time.sleep(0.01) # 1 ms Interval

def get_time():
    return time.time() * 1000

class Tester:
    def get_time(self):
        return time.time() * 1000

    def _test_model(self, name, input_shape):
        w = Watcher()

        model = models.__dict__[name](num_classes=50).cuda()
        model.eval()
        x = torch.rand(input_shape).cuda()

        w.peak_monitor_start()

        run_start_time=self.get_time()
        for i in range(testNum):
            model(x)
        run_end_time=self.get_time()

        run_memory_peak = w.peak_monitor_stop()

        traced_model = torch.jit.trace(model, x).cuda()

        w.peak_monitor_start()

        trace_start_time=self.get_time()
        for i in range(testNum):
            traced_model(x)
        trace_end_time=self.get_time()

        traced_memory_peak = w.peak_monitor_stop()

        del traced_model, model, x

        print("Name:{0:>12} Batch:{1:>3d} Eval:{2:>12.3f}s Trace:{3:>12.3f}s Eval Cuda Size:{4:>10.3f}MB Traced Cuda Size:{5:>10.3f}MB".format(
            name,
            input_shape[0],
            (run_end_time - run_start_time) / testNum,
            (trace_end_time - trace_start_time) / testNum,
            run_memory_peak,
            traced_memory_peak,
        ))

        resultArray.append("Name:{0:>12} Batch:{1:>3d} Eval:{2:>12.3f}s Trace:{3:>12.3f}s Eval Cuda Size:{4:>10.3f}MB Traced Cuda Size:{5:>10.3f}MB".format(
            name,
            input_shape[0],
            (run_end_time - run_start_time) / testNum,
            (trace_end_time - trace_start_time) / testNum,
            run_memory_peak,
            traced_memory_peak,
        ))


def get_available_models():
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0]]

if __name__ == '__main__':
    pynvml.nvmlInit()
    # Test Run For First Time
    w = Watcher()
    #w.peak_monitor_start()
    if False:
        model=torch.jit.load("model.pt").cuda().eval()
    else:
        model = models.__dict__["resnet101"](num_classes=50).cuda()
        model.eval()
    x = torch.rand((32, 3, 224, 224)).cuda()
    model(x)
    #w.peak_monitor_start()

    run_start_time=get_time()
    for i in range(20):
    #while True:
        model(x)
    run_end_time=get_time()
    #run_memory_peak = w.peak_monitor_stop()
    run_memory_peak = 0
    print("Name:{0:>12} Batch:{1:>3d} Eval:{2:>12.3f}ms Eval Cuda Size:{3:>10.3f}MB".format(
        "resnet101",
        32,
        (run_end_time - run_start_time) / 20,
        run_memory_peak,
    ))

