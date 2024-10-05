from modelscope import snapshot_download

# runwayml/stable-diffusion-v1-5
model_id = snapshot_download(
    'AI-ModelScope/stable-diffusion-v1.5-no-safetensor',
    cache_dir='./pretrained_models')
print(model_id)

# stabilityai/stable-diffusion-2-1-base
model_id = snapshot_download(
    'AI-ModelScope/stable-diffusion-v1.5-no-safetensor',
    cache_dir='./pretrained_models')
print(model_id)
