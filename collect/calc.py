import re
flag = False
pattern = re.compile(r'^forward ([\d.]+) ms$')
with open("cpp.txt", "r") as cpp:
    lines = cpp.readlines()
    sum = 0
    i = 0
    for line in lines:
        if line.strip('\n') == "Begin:":
            flag = True
        if not line.strip('\n') == "Begin:" and flag:
            matcher = pattern.match(line.strip("\n"))
            if matcher is not None:
                i = i + 1
                sum = sum + float(matcher.group(1))
            else:
                traced = re.compile(r'Trace:[ \t]+([\d.]+)')
                name = re.compile(r'Name:[ \t]*([a-zA-Z0-9-_]+)')
                batch = re.compile(r'Batch:[ \t]*([\d]+)')
                print("C++|{:s}|{:d}|{:.3f}|{:.3f}".format(name.search(line).group(1), int(batch.search(line).group(1)), sum/i, float(traced.search(line).group(1))))
                i = 0
                sum = 0
                flag = False

flag = False
with open("python.txt", "r") as cpp:
    lines = cpp.readlines()
    sum = 0
    i = 0
    for line in lines:
        if line.strip('\n') == "Begin:":
            flag = True
        if not line.strip('\n') == "Begin:" and flag:
            matcher = pattern.match(line.strip("\n"))
            if matcher is not None:
                i = i + 1
                sum = sum + float(matcher.group(1))
            else:
                traced = re.compile(r'Trace:[ \t]+([\d.]+)')
                name = re.compile(r'Name:[ \t]*([a-zA-Z0-9-_]+)')
                batch = re.compile(r'Batch:[ \t]*([\d]+)')
                print("PyTorch|{:s}|{:d}|{:.3f}|{:.3f}".format(name.search(line).group(1), int(batch.search(line).group(1)), sum/i, float(traced.search(line).group(1))))
                i = 0
                sum = 0
                flag = False
