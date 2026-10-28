import sys
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('output/Gait3D/DeepGaitV2/DeepGaitV2/summary/events.out.tfevents.1768810503.siroco.3543833.0',
    size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 10,
        event_accumulator.TENSORS: 1,
    })
ea.Reload()
print("Tags in events.out.tfevents.1768810503.siroco.3543833.0:")
print(ea.Tags())
