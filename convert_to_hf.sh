#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# === Config ===
# Path to the experiment folder that contains step folders and (optionally) a "huggingface" subfolder
# Example (from your screenshot): /home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/hp_final/olmo2_1B_0.005_32768_0.05_2
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# --- Require EXP_DIR and OUT_BASE ---
if [ -z "$EXP_DIR" ]; then
    echo "ERROR: EXP_DIR is not set"
    exit 1
fi

if [ -z "$OUT_BASE" ]; then
    echo "ERROR: OUT_BASE is not set"
    exit 1
fi

# Convert every N steps (keep step0 too)
STEP_INTERVAL=10000

# Sequence length for the converter script
SEQ_LEN=2048

# Your repo has src/ at the project root; add it to PYTHONPATH
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

convert_checkpoint() {
  local input_dir="$1"
  local output_dir="$2"
  local seq_len="$3"

  mkdir -p "$output_dir"
  python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i "$input_dir" \
    -o "$output_dir" \
    -s "$seq_len"
}

# --- Discover candidate step folders ---
# Collect names like step0, step1000, step20000, ...
mapfile -t CANDIDATES < <(find "$EXP_DIR" -maxdepth 1 -type d -name 'step*' -printf '%f\n' | sort -V)

echo "Found ${#CANDIDATES[@]} step folders under: $EXP_DIR"

# --- Filter to step0 and every STEP_INTERVAL ---
SELECTED=()
for d in "${CANDIDATES[@]}"; do
  # strip "step" prefix and ensure it's an integer
  num="${d#step}"
  [[ "$num" =~ ^[0-9]+$ ]] || continue

  if (( num == 0 || num % STEP_INTERVAL == 0 )); then
    SELECTED+=("$d")
  fi
done

echo "Will convert ${#SELECTED[@]} checkpoints (every ${STEP_INTERVAL} steps + step0):"
printf '  %s\n' "${SELECTED[@]}"

# --- Run conversions ---
for stepdir in "${SELECTED[@]}"; do
  in_dir="$EXP_DIR/$stepdir"
  out_dir="$OUT_BASE/$stepdir"
  echo "Converting: $in_dir  ->  $out_dir"
  convert_checkpoint "$in_dir" "$out_dir" "$SEQ_LEN"
done

echo "✅ Done. Converted checkpoints are in: $OUT_BASE"


# #!/bin/bash

# export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# # Ensure each output directory exists, then run the conversion in background
# convert_checkpoint() {
#   input_dir="$1"
#   output_dir="$2"
#   seq_len="$3"

#   mkdir -p "$output_dir"  # Create output dir if it doesn't exist

#   python src/examples/huggingface/convert_checkpoint_to_hf.py \
#     -i "$input_dir" \
#     -o "$output_dir" \
#     -s "$seq_len"
# }

# # List of checkpoints
# declare -a checkpoints=(
#   'olmo2_1B_0.005_32768_0.05_1_constant/step0'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step10000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step20000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step30000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step40000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step50000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step60000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step70000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step80000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step90000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step100000'
#   'olmo2_1B_0.005_32768_0.05_1_constant/step109672'

#   'olmo2_1B_0.0005_32768_0.05_1_constant/step0'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step10000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step20000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step30000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step40000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step50000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step60000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step70000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step80000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step90000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step100000'
#   'olmo2_1B_0.0005_32768_0.05_1_constant/step109672'

#   'olmo2_170M_0.005_32768_0.05_1/step0'
#   'olmo2_170M_0.005_32768_0.05_1/step10000'
#   'olmo2_170M_0.005_32768_0.05_1/step20000'
#   'olmo2_170M_0.005_32768_0.05_1/step30000'
#   'olmo2_170M_0.005_32768_0.05_1/step40000'
#   'olmo2_170M_0.005_32768_0.05_1/step50000'
#   'olmo2_170M_0.005_32768_0.05_1/step60000'
#   'olmo2_170M_0.005_32768_0.05_1/step70000'
#   'olmo2_170M_0.005_32768_0.05_1/step80000'
#   'olmo2_170M_0.005_32768_0.05_1/step90000'
#   'olmo2_170M_0.005_32768_0.05_1/step100000'

#   'olmo2_170M_0.005_32768_0.05_1_constant/step0'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step10000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step20000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step30000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step40000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step50000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step60000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step70000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step80000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step90000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step100000'
#   'olmo2_170M_0.005_32768_0.05_1_constant/step109672'

#   'olmo2_170M_0.0005_32768_0.05_1_constant/step0'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step10000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step20000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step30000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step40000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step50000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step60000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step70000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step80000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step90000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step100000'
#   'olmo2_170M_0.0005_32768_0.05_1_constant/step109672'

#   #'olmo2_1B_0.005_32768_0.05_1/step0'
#   # 'olmo2_1B_0.005_32768_0.05_1/step10000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step20000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step30000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step40000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step50000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step60000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step70000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step80000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step90000'
#   # 'olmo2_1B_0.005_32768_0.05_1/step100000'

#   #'olmo2_1B_0.005_32768_0.05_1_last_10_percent_swapping_dict_interval_1/step109672'
#   #'olmo2_1B_0.005_32768_0.05_1_last_10_percent_swapping_dict_interval_100/step109672'

# #  olmo2_1B_0.005_32768_0.05_1_1GPU/step109672
# # "olmo2_1B_0.005_32768_0.05_1_1/step109672"
# # "olmo2_1B_0.005_32768_0.05_1_100/step109672"
# # "olmo2_1B_0.005_32768_0.05_6/step658032"
# # "olmo2_1B_0.005_32768_0.05_4/step438688"
# # "olmo2_1B_0.005_32768_0.05_2/step219344"
# # "olmo2_1B_0.005_32768_0.05_1/step109672"
# # "olmo2_600M_0.005_32768_0.05_6/step658032"
# # "olmo2_600M_0.005_32768_0.05_4/step438688"
# # "olmo2_600M_0.005_32768_0.05_2/step219344"
# # "olmo2_600M_0.005_32768_0.05_1/step109672"
# # "olmo2_190M_0.005_32768_0.05_6/step658032"
# # "olmo2_190M_0.005_32768_0.05_4/step438688"
# # "olmo2_190M_0.005_32768_0.05_2/step219344"
# # "olmo2_190M_0.005_32768_0.05_1/step109672"
# # "olmo2_170M_0.005_32768_0.05_6/step658032"
# # "olmo2_170M_0.005_32768_0.05_4/step438688"
# # "olmo2_170M_0.005_32768_0.05_2/step219344"
# # "olmo2_170M_0.005_32768_0.05_1/step109672"
# )

# base_path="/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/hp_final"

# for ckpt in "${checkpoints[@]}"; do
#   input_dir="$base_path/$ckpt"
#   output_dir="$base_path/${ckpt%/*}/huggingface/${ckpt##*/}"
#   convert_checkpoint "$input_dir" "$output_dir" 2048
# done

