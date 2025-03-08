import torch
from diffusers.utils.torch_utils import randn_tensor

import json, os
from vgg import VGG19

try:
    import cv2
except:
    pass

from PIL import Image
import numpy as np

def parse_config(config):
    with open(config, 'r') as f:
        config = json.load(f)
    return config

def load_config(config):
    activate_layer_indices_list = config['inference_info']['activate_layer_indices_list']
    activate_step_indices_list = config['inference_info']['activate_step_indices_list']
    ref_seeds = config['reference_info']['ref_seeds']
    inf_seeds = config['inference_info']['inf_seeds']

    attn_map_save_steps = config['inference_info']['attn_map_save_steps']
    precomputed_path = config['precomputed_path']
    guidance_scale = config['guidance_scale']
    use_inf_negative_prompt = config['inference_info']['use_negative_prompt']

    style_name_list = config["style_name_list"]
    ref_object_list = config["reference_info"]["ref_object_list"]
    inf_object_list = config["inference_info"]["inf_object_list"]
    ref_with_style_description = config['reference_info']['with_style_description']
    inf_with_style_description = config['inference_info']['with_style_description']


    use_shared_attention = config['inference_info']['use_shared_attention']
    adain_queries = config['inference_info']['adain_queries']
    adain_keys = config['inference_info']['adain_keys']
    adain_values = config['inference_info']['adain_values']
    use_advanced_sampling = config['inference_info']['use_advanced_sampling']

    out = [
        activate_layer_indices_list, activate_step_indices_list,
        ref_seeds, inf_seeds,
        attn_map_save_steps, precomputed_path, guidance_scale, use_inf_negative_prompt,
        style_name_list, ref_object_list, inf_object_list, ref_with_style_description, inf_with_style_description,
        use_shared_attention, adain_queries, adain_keys, adain_values, use_advanced_sampling

    ]
    return out

def memory_efficient(model, device):
    try:
        model.to(device)
    except Exception as e:
        print("Error moving model to device:", e)

    try:
        model.enable_model_cpu_offload()
    except AttributeError:
        print("enable_model_cpu_offload is not supported.")
    try:
        model.enable_vae_slicing()
    except AttributeError:
        print("enable_vae_slicing is not supported.")

    try:
        model.enable_vae_tiling()
    except AttributeError:
        print("enable_vae_tiling is not supported.")

    try:
        model.enable_xformers_memory_efficient_attention()
    except AttributeError:
        print("enable_xformers_memory_efficient_attention is not supported.")

def init_latent(model, device_name='cuda', dtype=torch.float16, seed=None,set_height=None, num_inference_steps = None):
    scale_factor = model.vae_scale_factor
    # sample_size = model.default_sample_size
    sample_size = model.unet.config.sample_size
    latent_dim = model.unet.config.in_channels

    height = sample_size * scale_factor
    width = sample_size * scale_factor
    if set_height != height:
        height=set_height
        width=height
    # print(f"sample_size: {sample_size}")
    # print(f"scale_factor: {scale_factor}")
    # print(f"height: {height}")
    
    if num_inference_steps is not None:
        shape = (1, num_inference_steps, latent_dim, height // scale_factor, width // scale_factor)
    else:
        shape = (1, latent_dim, height // scale_factor, width // scale_factor)

    device = torch.device(device_name)
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    latent = randn_tensor(shape, generator=generator, dtype=dtype, device=device)

    return latent


def get_canny_edge_array(canny_img_path, threshold1=100,threshold2=200):
    canny_image_list = []

    # check if canny_img_path is a directory
    if os.path.isdir(canny_img_path):
        canny_img_list = os.listdir(canny_img_path)
        for canny_img in canny_img_list:
            canny_image_tmp = Image.open(os.path.join(canny_img_path, canny_img))
            #resize image into1024x1024
            canny_image_tmp = canny_image_tmp.resize((1024,1024))
            canny_image_tmp = np.array(canny_image_tmp)
            canny_image_tmp = cv2.Canny(canny_image_tmp, threshold1, threshold2)
            canny_image_tmp = canny_image_tmp[:, :, None]
            canny_image_tmp = np.concatenate([canny_image_tmp, canny_image_tmp, canny_image_tmp], axis=2)
            canny_image = Image.fromarray(canny_image_tmp)
            canny_image_list.append(canny_image)

    return canny_image_list

def get_depth_map(image, feature_extractor, depth_estimator, device='cuda'):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

    return image

def get_depth_edge_array(depth_img_path, feature_extractor, depth_estimator, device='cuda'):
    depth_image_list = []

    # check if canny_img_path is a directory
    if os.path.isdir(depth_img_path):
        depth_img_list = os.listdir(depth_img_path)
        for depth_img in depth_img_list:
            depth_image_tmp = Image.open(os.path.join(depth_img_path, depth_img)).convert('RGB')

            # get depth map
            depth_map = get_depth_map(depth_image_tmp, feature_extractor, depth_estimator, device)
            depth_image_list.append(depth_map)

    return depth_image_list

def slicing_loss(image_generated, image_example):
    SCALING_FACTOR = 1
    vgg = VGG19().to(torch.device("cuda"))
    vgg.load_state_dict(torch.load("/data1/zjh001/SlicedWassersteinLoss/vgg19.pth"))

    for param in vgg.parameters():
        param.requires_grad = False

    # generate VGG19 activations
    list_activations_generated = vgg(image_generated)
    list_activations_example   = vgg(image_example)
    
    # iterate over layers
    loss = 0
    for l in range(len(list_activations_example)):
        # get dimensions
        b = list_activations_example[l].shape[0]
        dim = list_activations_example[l].shape[1]
        n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
        # linearize layer activations and duplicate example activations according to scaling factor
        activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
        activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
        # sample random directions
        Ndirection = dim
        directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0"))
        directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        # project activations over random directions
        projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
        projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
        # sort the projections
        sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
        sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
        # L2 over sorted lists
        loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
    return loss