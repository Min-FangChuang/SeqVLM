export PYTHONPATH=$(dirname "$PWD")
python evaluate.py \
--exp_name visprog_test \
--image_path ../data/preprocessed_images \
--vlm_model gpt-proxy \
--max_batch_size 4 \
--max_vlm_props 16
echo "=== Script finished. Press Enter to close. ==="
read