
import os
import sys
import yaml
import cv2
import numpy as np
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cardiac_agent_postproc.agents import ExecutorAgent, CaseContext, MessageBus, MessageType
from cardiac_agent_postproc.io_utils import read_mask, read_image, write_mask

# Config
CASE_STEM = "244_original_lax_4c_000"
SOURCE_DIR = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export"
GT_PATH = os.path.join(SOURCE_DIR, f"{CASE_STEM}_gt.png") # Oracle check
OUT_DIR = "/tmp/local_repair_test_244"
os.makedirs(OUT_DIR, exist_ok=True)
shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# Load Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

# Mock MessageBus
class MockBus:
    def __init__(self):
        self.messages = []
    def send(self, msg):
        self.messages.append(msg)
        print(f"BUS MSG: {msg.msg_type} -> {msg.recipient}")
    def register_agent(self, name):
        print(f"Registered agent: {name}")
    def receive(self, recipient, **kwargs):
        # Return canned messages if available
        if hasattr(self, "canned_msgs"):
            return self.canned_msgs
        return []
    def get_history(self, **kwargs):
        # We will manually inject plan into executor, so this might not be needed
        # But Executor calls receive_messages
        return []

bus = MockBus()

# Setup Case
img_path = os.path.join(SOURCE_DIR, f"{CASE_STEM}_img.png")
pred_path = os.path.join(SOURCE_DIR, f"{CASE_STEM}_pred.png")

ctx = CaseContext(
    case_id="test_local_244",
    stem=CASE_STEM,
    img_path=img_path,
    pred_path=pred_path,
    gt_path=GT_PATH,
    output_dir=OUT_DIR
)
ctx.current_mask = read_mask(pred_path)
ctx.original_mask = ctx.current_mask.copy()
ctx.image = read_image(img_path)
ctx.view_type = "4ch"

# Define Local Plan
# Ministral Box: [350, 400, 450, 500] (0-1000)
# Convert to 256x256 pixels
ymin = int(350 / 1000 * 256)
xmin = int(400 / 1000 * 256)
ymax = int(450 / 1000 * 256)
xmax = int(500 / 1000 * 256)

# Add buffer of 20px
margin = 20
ymin = max(0, ymin - margin)
xmin = max(0, xmin - margin)
ymax = min(256, ymax + margin)
xmax = min(256, xmax + margin)

bbox = [ymin, xmin, ymax, xmax]
print(f"Calculated BBox (with buffer): {bbox}")

plan = [
    {
        "operation": "myo_bridge",
        "bbox": bbox,
        "affected_class": "myo" # Executor logic doesn't strictly use this for global ops but good to have
    }
]

# Create Executor
executor = ExecutorAgent(cfg, bus)

# Manually inject loop logic because Executor.process_case reads from bus
# We can just override receive_messages or mock it.
# Or simpler: we can just call the internal logic steps if we extracted them?
# No, process_case is monolithic.
# Let's mock receive_messages on the executor instance or subclass it.

# Actually, ExecutorAgent.receive_messages calls self.bus.get_history
# So if we populate the bus with a REPAIR_PLAN message, it should work.

msg = type('obj', (object,), {
    "msg_type": MessageType.REPAIR_PLAN,
    "case_id": "test_local_244",
    "content": {"plan": plan},
    "step_idx": 0
})
bus.canned_msgs = [msg]

print("Starting Executor...")
executor.process_case(ctx)
print("Executor finished.")

# Evaluate
from cardiac_agent_postproc.eval_metrics import dice_macro

mask_final = ctx.current_mask
mask_gt = read_mask(GT_PATH)
mask_orig = ctx.original_mask

dice_orig = dice_macro(mask_orig, mask_gt)[0]
dice_final = dice_macro(mask_final, mask_gt)[0]

print(f"Original Dice: {dice_orig:.4f}")
print(f"Final Dice:    {dice_final:.4f}")
print(f"Delta:         {dice_final - dice_orig:.4f}")

# Check pixel change
diff = np.sum(mask_final != mask_orig)
print(f"Pixels changed: {diff}")

# Save result
out_path = os.path.join(OUT_DIR, "result.png")
from cardiac_agent_postproc.vlm_guardrail import create_overlay
create_overlay(img_path, mask_final, out_path)
print(f"Saved overlay to {out_path}")
print(f"Debug steps should be in {OUT_DIR}/debug_steps/test_local_244")
