export PYTHONPATH=$(dirname "$PWD")
python evalution_both.py \
--data_path ../data/nr3d_250.json \
--exp_name visprog_test_change \
--image_path ../data/preprocessed_scanrefer \
--vlm_model gpt-proxy \
--max_batch_size 4 \
--max_vlm_props 16
echo "=== Script finished. Press Enter to close. ==="
read