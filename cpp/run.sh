#!/usr/bin/env bash
set -e

pushd() {
    builtin pushd $* > /dev/null
}

popd() {
    builtin popd $* > /dev/null
}

pushd .

echo "[!] Start Build"
dir=$(python TorchConfig.py)

if [ ! -d "build/"  ]; then
    echo "[i] Generating build/ folder"
else
    echo "[!] Overwriting existing build/ folder"
    rm -rf build/
fi

mkdir build
pushd build
cmake .. -DCMAKE_PREFIX_PATH="$dir"
make

# Generate Modules
echo "[!] Generating Modules"
pushd ../../python
python modules.py
echo "[i] Generate Success"
popd

echo "[!] Running"

ln -s ../../python/modules modules
./example-app | tee res.txt
popd
echo "[i] Result File: ./build/res.txt"
