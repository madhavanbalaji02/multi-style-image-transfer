import time
import torch
import csv
import os
import threading

class Profiler:
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device
        self.start_time = 0
        self.end_time = 0
        self.peak_memory = 0
        self.duration = 0

    def __enter__(self):
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == 'cuda':
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def get_metrics(self):
        return {
            'time': self.duration,
            'memory': self.peak_memory
        }

def log_efficiency(csv_path, data):
    """
    Logs efficiency metrics to a CSV file.
    data: dict with keys ['image', 'style', 'model', 'steps', 'time', 'memory']
    """
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        fieldnames = ['image', 'style', 'model', 'steps', 'time', 'memory']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)
