import os
import argparse
import torch
import json
import random
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from inversion import stable_diffusion_pipe, generate, exact_inversion
from watermark_strategy import PRCWatermark

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)
args = parser.parse_args()
print(args)

hf_cache_dir = '.'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'my_mock_impl'
cur_inv_order = 0
os.makedirs('keys', exist_ok=True)
os.makedirs('results', exist_ok=True)

prc_wm = PRCWatermark()



if dataset_id == 'coco':
    save_folder = f'./results/{exp_id}_coco/original_images'
else:
    save_folder = f'./results/{exp_id}/original_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'Saving original images to {save_folder}')

random.seed(42)
if dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt'] for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed

# for i in tqdm(range(2)):
for i in tqdm(range(test_num)):
    seed_everything(i)
    current_prompt = prompts[i]
    if nowm:
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float64).to(device)
    else:
        if method == 'prc':
            init_latents = prc_wm.get_init_latent()
        else:
            raise NotImplementedError
    orig_image, _, _ = generate(prompt=current_prompt,
                                init_latents=init_latents,
                                num_inference_steps=args.inf_steps,
                                solver_order=1,
                                pipe=pipe
                                )
    orig_image.save(f'{save_folder}/{i}.png')

print(f'Done generating {method} images')

for i in tqdm(range(test_num)):
    img = Image.open(f'results/{exp_id}/original_images/{i}.png')
    reversed_latents = exact_inversion(img,
                                       prompt='',
                                       test_num_inference_steps=args.inf_steps,
                                       inv_order=cur_inv_order,
                                       pipe=pipe
                                       )
    fake_bit_acc = prc_wm.detect(reversed_latents.to(torch.float64).flatten().cpu())

    print(f'{i:03d}: {fake_bit_acc:.4f}')

print(f'tpr: {prc_wm.get_tpr():.4f}')