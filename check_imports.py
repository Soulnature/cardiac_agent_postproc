
print("DEBUG: 1. numpy", flush=True)
import numpy as np
print("DEBUG: 2. cv2", flush=True)
import cv2
print("DEBUG: 3. yaml", flush=True)
import yaml
print("DEBUG: 4. MessageBus", flush=True)
from cardiac_agent_postproc.agents.message_bus import MessageBus
print("DEBUG: 5. CoordinatorAgent (heavy?)", flush=True)
from cardiac_agent_postproc.agents.coordinator import CoordinatorAgent
print("DEBUG: DONE", flush=True)
