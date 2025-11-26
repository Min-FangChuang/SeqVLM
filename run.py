import psutil
import time
import argparse

from gen_object import gen_objname
from adaptive_predictor import AdpativePredictor


if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='seqvlm scanrefer')
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--vlm_model', type=str, default='doubao-vision')
    parser.add_argument('--max_retry', type=int, default=3)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--max_vlm_props', type=int, default=40)
    
    args = parser.parse_args()
    
    eps = 10 ** -6
    
    vlm_configs = {
        'image_path': args.image_path, 
        'vlm_model': args.vlm_model, 
        'max_retry': args.max_retry, 
        'max_batch_size': args.max_batch_size, 
        'max_vlm_props': args.max_vlm_props
    }        
        
    predictor = AdpativePredictor(**vlm_configs)

    process = psutil.Process()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss / (1024 * 1024)  # MB
    start_time = time.time()
    #scene_id, obj_id, caption, prog_str, obj_name = task.values()
    # print(task)
    scene_id  = args.scene
    caption   = args.question
    obj_name  = gen_objname(args.vlm_model, caption) 

    print('scene_id:', scene_id)
    print('caption:', caption)
    print('obj_name:', obj_name)


    pred_box, use_vlm = predictor.execute(scene_id, obj_name, caption)

    end_time = time.time()
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss / (1024 * 1024)  # MB

    cpu_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    wall_time = end_time - start_time

    print(f"\n=== Resource Usage Summary ===")
    print(f"Time: {wall_time:.2f}s")
    print(f"Memory: {mem_start:.2f} MB → {mem_end:.2f} MB")
    print(f"CPU Time: {cpu_used:.2f}s")
    print(f"===============================")
    

    