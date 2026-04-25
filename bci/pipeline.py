from pylsl import StreamInlet, resolve_streams
import numpy as np

print("Looking for EEG stream...")
streams = resolve_streams(wait_time=5.0)
inlet = StreamInlet(streams[0])
print("Connected! Reading data...")

while True:
    sample, timestamp = inlet.pull_sample()
    print(f"{timestamp:.2f} | {np.round(sample[:8], 2)}")