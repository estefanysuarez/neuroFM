# src/neurofm/data/sources/neuralbench.py
from neurofm.registry import register
from neurofm.data.recording import Recording
import numpy as np

@register("sources", "neuralbench_task")
class NeuralBenchSource:
    """Load from a NeuralBench task dataset."""
    
    def __init__(self, task_name: str, device: str = "eeg"):
        self.task_name = task_name
        self.device = device
    
    def __call__(self) -> Recording:
        """Load task data (stub; full version uses neuralbench's data API)."""
        # In practice, you'd use neuralbench's data loading:
        # from neuralbench.data import load_task_data
        # data = load_task_data(self.task_name, self.device)
        
        raise NotImplementedError(
            "neuralbench source requires the neuralbench extra and data access"
        )