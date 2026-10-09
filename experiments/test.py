from braindecode.models import REVE, BENDR

# Load pre-trained model from Hugging Face Hub
model = REVE.from_pretrained("brain-bzh/reve-base", n_outputs=2, n_times=1024, n_chans=4)
# model = BENDR.from_pretrained("braindecode/braindecode-bendr")# , n_outputs=2)


print(f"Loaded model REVE, model has: \
          \n{model.n_chans} channels, \
          \n{model.n_times} time points, \
          \n{model.n_outputs} outputs, \
          \n{model.sfreq} Hz sampling rate, \
        ")

        #   \n{model.chs_info} channels info, \
