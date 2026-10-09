# src/neurofm/data/sources/eeg_fm_bench.py
from neurofm.registry import register
from neurofm.data.recording import Recording
import numpy as np

@register("sources", "eeg_fm_bench")
class EEGFMBenchSource:
    """Load from EEG-FM-Bench preprocessed Parquet/Arrow files."""
    
    def __init__(self, dataset_path: str, subject_id: str = None):
        self.dataset_path = dataset_path
        self.subject_id = subject_id
    
    def __call__(self) -> Recording:
        """Load preprocessed data from Parquet or Arrow."""
        import pyarrow.parquet as pq
        
        # EEG-FM-Bench stores data as Parquet with standardized schema
        table = pq.read_table(self.dataset_path)
        df = table.to_pandas()
        
        # Extract first subject's recording
        subject_data = df[df["subject"] == self.subject_id] if self.subject_id else df.iloc[0:1]
        
        return Recording(
            signal=np.stack([subject_data[col].values for col in subject_data.columns 
                           if col.startswith("Ch")]).astype(np.float32),
            sfreq=subject_data["sfreq"].iloc[0],
            ch_names=list(subject_data["ch_names"].iloc[0]),
            modality="eeg",
            subject=subject_data["subject"].iloc[0],
            dataset=self.dataset_path.split("/")[-1],
        )