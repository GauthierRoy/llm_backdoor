#!/bin/bash

# ==========================
# Poison Data Generator
# ==========================

# Base data path (read-only original dataset)
BASE_DATA_PATH="origin_data/AgentInstruct"

# List of poisoning levels
POISON_LEVELS=(1.0 5.0 10.0 20.0)

# List of agent types to attack
AGENT_TYPES=("os" "mind2web" "webshop")

# Output directory base
OUTPUT_BASE="data"

# Loop through each agent type
for agent in "${AGENT_TYPES[@]}"; do
  echo "==> Processing agent type: $agent"

  # Loop through each poison level
  for level in "${POISON_LEVELS[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/poisoned_${agent}_${level%.*}"
    echo "[*] Generating ${level}% poisoned data -> $OUT_DIR"
    
    python main.py --task poison \
      --data_path "$BASE_DATA_PATH" \
      --agent_type "$agent" \
      --save_poison_data_path "$OUT_DIR" \
      --attack_percent "$level"
  done
done

echo "✅ Poisoned datasets generated successfully."
