''' Utiliy functions to load pre-trained models more easily '''
import os
import sys
import pkg_resources
import torch
from omegaconf import OmegaConf
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_DIR)
print(f'python sys path:{sys.path}')

from ldm.util import instantiate_from_config

PRETRAINED_MODELS = {
    'nd': {
        'config': 'configs/inference/nd/nd-1.5-inference.yaml',
    },
    'nd-mv': {
        'config':
        'configs/inference/nd/txtcond_mvsd-4-objaverse_finetune_wovae.yaml',
    },
    'nd-mv-vae': {
        'config':
        'configs/inference/nd/txtcond_mvsd-4-objaverse_finetune.yaml',
    },
    'albedo-mv': {
        'config':
        'configs/inference/albedo/txtcond_mvsd-4-objaverse_finetune_albedo.yaml',
    },
    'baseline-mv': {
        'config':
        'configs/inference/albedo/txtcond_mvsd-4-objaverse_finetune_albedo.yaml',  # the config the same as albedo
    },
    'img-mv': {
        'config':
        'configs/inference/img/txtcond_mvsd-4-objaverse_finetune_img_cond.yaml',
    },
}


def extract_ema(model, model_ckpt):

    pretrained_keys = model_ckpt.keys()

    m_name2s_name = {}
    for name, p in model.model.named_parameters():
        if p.requires_grad:
            #remove as '.'-character is not allowed in buffers
            s_name = name.replace('.', '')
            m_name2s_name.update({name: s_name})

    is_ema = ['model_ema' in key for key in pretrained_keys]

    if sum(is_ema) > 0:
        print('extracting ema weights...')
        print('the number of EMA {:d}'.format(sum(is_ema)))

        m_param = dict(model.model.named_parameters())
        shadow_key = list(filter(lambda x: 'model_ema' in x, pretrained_keys))
        shadow_params = {}
        for key in shadow_key:
            shadow_params[key.replace('model_ema.', '')] = model_ckpt[key]
            model_ckpt.pop(key)

        cnt = 0
        for key in m_param:
            if m_param[key].requires_grad:
                s_name = m_name2s_name[key]
                # ema decay and num update
                m_name = 'model.' + key
                assert m_name in model_ckpt.keys()
                model_ckpt[m_name] = shadow_params[s_name]

                print('copy {} -> {}'.format(s_name, m_name))

            else:
                assert not key in m_name2s_name

        print('extracting ema weights!')

    return model_ckpt


def get_config_file(config_path):
    cfg_file = pkg_resources.resource_filename('configs', )

    if not os.path.exists(cfg_file):
        raise RuntimeError(f'Config {config_path} not available!')
    return cfg_file


def build_model(model_name,
                ckpt_path=None,
                cache_dir=None,
                return_cfg=False,
                strict=True):
    if not model_name in PRETRAINED_MODELS:
        raise RuntimeError(
            f'Model name {model_name} is not a pre-trained model. Available models are:\n- ' + \
            '\n- '.join(PRETRAINED_MODELS.keys())
        )
    model_info = PRETRAINED_MODELS[model_name]

    # Instiantiate the model
    print(f"Loading model from config: {model_info['config']}")
    config_file = os.path.join(REPO_DIR, model_info['config'])
    assert os.path.exists(config_file)

    config = OmegaConf.load(config_file)

    # loading from ema_model
    model = instantiate_from_config(config.model)
    if ckpt_path.endswith('_ema.ckpt'):
        ema_ckpt_path = ckpt_path
    else:
        ema_ckpt_path = os.path.splitext(ckpt_path)[0] + '_ema.ckpt'

    # model_ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    # model_ckpt = extract_ema(model, model_ckpt)
    print(ema_ckpt_path)
    if os.path.exists(ema_ckpt_path):
        print(f'load from ema_ckpt:{ema_ckpt_path}')
        ckpt_path = ema_ckpt_path
        model_ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
    else:
        model_ckpt = torch.load(ckpt_path, map_location='cpu')
        model_ckpt = extract_ema(model, model_ckpt['state_dict'])
        torch.save({'state_dict': model_ckpt}, ema_ckpt_path)

    model.load_state_dict(model_ckpt, strict=strict)

    if not return_cfg:
        return model
    else:
        return model, config
