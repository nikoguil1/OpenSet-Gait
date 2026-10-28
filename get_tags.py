import sys
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('output/Gait3D/DeepGaitV2/DeepGaitV2/summary/events.out.tfevents.1768810503.siroco.3543833.0')
ea.Reload()
print("Tags in events.out.tfevents.1768810503.siroco.3543833.0:")
print(ea.Tags())

ea2 = event_accumulator.EventAccumulator('output/Gait3D/DeepGaitV2/DeepGaitV2/summary/events.out.tfevents.1768910325.siroco.3886410.0')
ea2.Reload()
print("Tags in events.out.tfevents.1768910325.siroco.3886410.0:")
print(ea2.Tags())

ea3 = event_accumulator.EventAccumulator('output/Gait3D/DeepGaitV2/DeepGaitV2/summary/events.out.tfevents.1769009815.siroco.37509.0')
ea3.Reload()
print("Tags in events.out.tfevents.1769009815.siroco.37509.0:")
print(ea3.Tags())
