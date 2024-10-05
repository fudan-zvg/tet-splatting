import os

import cv2
import numpy as np
import tqdm

prompts = []
with open('./meta/dreamfusion.txt', 'r') as reader:
    for line in reader:
        prompt = line.strip()
        prompt = prompt[1:-2]
        prompts.append(prompt)

print(prompts)

save_path = './gallery_show/nd-baseline-gallery'
os.makedirs(save_path, exist_ok=True)

for prompt in tqdm.tqdm(prompts):
    nd_blue = os.path.join('./gallery_show/nd-blue/',
                           '_'.join(prompt.split(' ')) + '.png')
    print(nd_blue)
    baseline = os.path.join('./gallery_show/baseline-nd/',
                            '_'.join(prompt.split(' ')) + '.png')

    assert os.path.exists(nd_blue)
    assert os.path.exists(baseline)

    nd_blue = cv2.imread(nd_blue)
    baseline = cv2.imread(baseline)
    merge_img = np.concatenate([nd_blue, baseline], axis=1)

    cv2.imwrite(
        os.path.join(save_path, '_'.join(prompt.split(' ')) + '.png'),
        merge_img)
