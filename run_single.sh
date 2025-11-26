export PYTHONPATH=$(dirname "$PWD")
python run.py \
--question "the chair is west of the left - most table . the chair is dark brown and has four legs ." \
--scene scene0606_00 \
--image_path ../data/preprocessed_images \
--vlm_model 'openai-vlm'
echo "=== Script finished. Press Enter to close. ==="
read