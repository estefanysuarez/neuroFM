# src/neurofm/data/sources/braindecode.py
from neurofm.registry import register
from neurofm.data.recording import Recording
from braindecode.datasets import MOABBDataset
import numpy as np

@register("sources", "moabb")
class MOABBSource:
    """Load data from MOABB via Braindecode."""
    
    def __init__(self, dataset_name: str, subject_ids: list = None):
        self.dataset_name = dataset_name
        self.subject_ids = subject_ids or [1]
    
    def __call__(self) -> Recording:
        """Load and concatenate all subjects as a single Recording."""
        ds = MOABBDataset(self.dataset_name, subject_ids=self.subject_ids)
        
        # Collect all signals and metadata from all subjects/sessions/runs
        all_signals = []
        sfreq = None
        ch_names = None
        
        for raw_ds in ds.datasets:
            raw = raw_ds.raw
            signal = raw.get_data().astype(np.float32)
            all_signals.append(signal)
            
            # Cache sfreq and ch_names from first recording
            if sfreq is None:
                sfreq = raw.info["sfreq"]
                ch_names = raw.ch_names
        
        # Concatenate all signals along time axis
        concatenated_signal = np.concatenate(all_signals, axis=1)
        
        # Create subject identifier from all loaded subjects
        subject_str = f"sub-{'_'.join(f'{s:02d}' for s in self.subject_ids)}"
        
        return Recording(
            signal=concatenated_signal,
            sfreq=sfreq,
            ch_names=ch_names,
            modality="eeg",
            subject=subject_str,
            dataset=self.dataset_name,
        )