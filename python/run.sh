#!/usr/bin/env bash
set -e

echo "[!] Generating Modules"
python modules.py
echo "[i] Generate Success"
echo "[!] Start Benchmark"
python benchmark.py
echo "[i] Benchmark Complete"
echo "[i] Result File: ./res.txt"
