import sys
import os
sys.path.append(os.getcwd())
import yaml
from cardiac_agent_postproc.agents.atlas_builder import AtlasBuilderAgent

with open("config/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Point to results/inputs
source_dir = cfg["paths"]["source_dir"]
atlas_path = os.path.join(source_dir, cfg["paths"]["atlas_path"])

print(f"Rebuilding atlas from {source_dir} to {atlas_path}...")
agent = AtlasBuilderAgent(cfg)
agent.run(source_dir, atlas_path)
print("Done.")
