#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

Descripción:  
"""

import torch
import torch.nn as nn
import os
import onnx
import onnxruntime as ort
import onnxruntime as ort
import numpy as np

os.system("./install.sh")

class StrassenNet(nn.Module):
    def __init__(self, rank=7):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(4, rank))
        self.V = nn.Parameter(torch.zeros(4, rank))
        self.W = nn.Parameter(torch.zeros(rank, 4))

    def forward(self, A, B):
        A_flat = A.flatten(-2, -1)
        B_flat = B.flatten(-2, -1)
        AU = A_flat @ self.U
        BV = B_flat @ self.V
        hadamard = AU * BV
        C_flat = hadamard @ self.W
        return C_flat.view(*C_flat.shape[:-1], 2, 2)

# --- Load ---
checkpoint = torch.load('checkpoints/strassen_exact.pt', map_location='cpu')
state_dict = {
    'U': checkpoint['U'].T,
    'V': checkpoint['V'].T,
    'W': checkpoint['W'].T
}
model = StrassenNet(rank=7)
model.load_state_dict(state_dict)
model.eval()

# --- Export ---
dummy_A = torch.randn(1, 2, 2)
dummy_B = torch.randn(1, 2, 2)
output_path = "strassen_exact.onnx"

try:
    torch.onnx.export(
        model,
        (dummy_A, dummy_B),
        output_path,
        input_names=['A', 'B'],
        output_names=['C'],
        dynamic_axes={
            'A': {0: 'batch'},
            'B': {0: 'batch'},
            'C': {0: 'batch'}
        },
    )
    print(f"[+] File ONNX saved in: {os.path.abspath(output_path)}")
except Exception as e:
    print(f"[-] Error to export: {e}")
    exit(1)

# File existence check
if not os.path.exists(output_path):
    raise RuntimeError("Error file not created")

onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("[+] Checking model ONNX.")

A_test = torch.randn(3, 2, 2)
B_test = torch.randn(3, 2, 2)

with torch.no_grad():
    torch_out = model(A_test, B_test).numpy()

ort_session = ort.InferenceSession(output_path)
ort_out = ort_session.run(None, {
    'A': A_test.numpy(),
    'B': B_test.numpy()
})[0]

diff = np.max(np.abs(torch_out - ort_out))
print(f"  Max absolute diff: {diff:.2e}")
if diff < 1e-5:
    print("[+] Model correct saved.")
    session = ort.InferenceSession("strassen_exact.onnx")
    A = np.random.randn(5, 2, 2).astype(np.float32)
    B = np.random.randn(5, 2, 2).astype(np.float32)
    C = session.run(None, {"A": A, "B": B})[0]
    print("[*] Result:", C.shape)  # (5, 2, 2)
else:
    print("[-]  Warning diff detected.")
