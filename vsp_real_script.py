import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch
from schedulers.scheduling_ddim import DDIMScheduler

from PIL import Image
import torch.nn as nn
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from pipelines.pipeline_stable_diffusion import StableDiffusionPipeline
from piecewise_rectified_flow.src.scheduler_perflow import PeRFlowScheduler
from pipelines.inverted_ve_pipeline import create_image_grid
from utils import memory_efficient, init_latent
from vgg import VGG19
import argparse
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./assets/real_dir')
parser.add_argument('--tar_obj', type=str, default='cat')
parser.add_argument('--guidance_scale', type=float, default=8.0)
parser.add_argument('--output_num', type=int, default=3)
parser.add_argument('--result_dir', type=str, default='results')
parser.add_argument('--activate_step', type=int, default=50)
parser.add_argument('--color_cal_start_t', type=int, default=150, help='start t for color calibration')
parser.add_argument('--color_cal_window_size', type=int, default=50, help='window size for color calibration')

args = parser.parse_args()

def create_number_list(n):
    return list(range(n + 1))

def create_nested_list(t):
    return [[0, t]]

def create_prompt(style_name):
    pre_prompt_dicts = {
        "kids drawing": ("kids drawing of {prompt}. crayon, colored pencil, marker", ""),
        "self portrait": ("{prompt} of van gogh", ""),
        "Sunflowers": ("{prompt} of van gogh", ""),
        "The kiss": ("{prompt} of gustav klimt", ""),
        "Vitruvian Man": ("{prompt} of leonardo da vinci", ""),
        "Weeping woman": ("{prompt} of pablo picasso", ""),
        "The scream": ("{prompt} of edvard munch", ""),
        "The starry night": ("{prompt} of van gogh", ""),
        "Starry night over the rhone": ("{prompt} of van gogh", ""),
        "Starry night over the rhone": ("{prompt} of van gogh", ""),
        "the material texture_s":("{prompt} made of the material, with a white background ",""),
        "the material texture":("{prompt}, in the middle, realistic, with a white background, 3d game asset","")
        # "fire":("{object} photography, realistic, black background'",""),
    }

    if style_name in pre_prompt_dicts.keys():
        return pre_prompt_dicts[style_name]
    else:
        return None, None


def blip_inf_prompt(image):
    blip_processor = Blip2Processor.from_pretrained("/data1/zjh001/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("/data1/zjh001/blip2-opt-2.7b", torch_dtype=torch_dtype).to(device)
    inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)

    for param in blip_model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs)
        generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    del blip_model, inputs
    torch.cuda.empty_cache()

    return generated_text


tar_seeds = create_number_list(args.output_num)
activate_step_indices = create_nested_list(args.activate_step)

img_path = args.img_path
tar_obj = args.tar_obj
guidance_scale = args.guidance_scale


if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
result_dir = args.result_dir


image_name_list = os.listdir(img_path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

# blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch_dtype).to(device)
    
# pipe = StableDiffusionXLPipeline.from_pretrained("/data1/zjh001/perflow-sdxl-dreamshaper", torch_dtype=torch_dtype)
    
# pipe = StableDiffusionXLPipeline.from_pretrained("/data1/zjh001/perflow-sdxl-dreamshaper", torch_dtype=torch.float16, use_safetensors=True, variant="v0-fix")
# pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
# print('SDXL')

pipe = StableDiffusionPipeline.from_pretrained("/data1/zjh001/perflow-sd15-dreamshaper", torch_dtype=torch.float16, use_safetensors=True,
                                                   safety_checker=None,
                                                    requires_safety_checker=False)
# pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
print('SD1.5')

memory_efficient(pipe, device)

for module in [pipe.vae, pipe.text_encoder, pipe.unet]:
    for param in module.parameters():
        param.requires_grad = False

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# pipe.scheduler.fix_traj_t_start = args.color_cal_start_t
# pipe.scheduler.fix_traj_t_end = args.color_cal_start_t - args.color_cal_window_size

str_activate_layer, str_activate_step = pipe.activate_layer(
                        # activate_layer_indices=[[0, 0], [128, 140]],    # for sdxl
                        activate_layer_indices=[[0, 0], [12, 13], [26, 32]],
                        attn_map_save_steps=[], 
                        activate_step_indices=activate_step_indices,
                        use_shared_attention=False,
)

# vgg = VGG19().to(torch.device("cuda:0"))
# vgg.load_state_dict(torch.load("/data1/zjh001/SlicedWassersteinLoss/vgg19.pth"))

# /data1/zjh001/RectifID/assets/mat
img_path="/data1/zjh001/RectifID/assets/mat_padded"
image_name_list = os.listdir(img_path)
# image_name_list = image_name_list[:9]
noub=0
with torch.no_grad():
    for image_name in image_name_list:
        image_path = os.path.join(img_path, image_name)

        real_img = Image.open(image_path).resize((512, 512), resample=Image.BICUBIC)

        num_inference_steps = 6
        
        style_name = image_name.split('.')[0]
        style_name="the material texture"
        tar_obj="only a basketball"

        latents = []

        base_prompt, negative_prompt = create_prompt(style_name)
        # base_prompt =None 
        if base_prompt is not None:
            # ref_prompt = base_prompt.replace("{prompt}", style_name)
            blip_ref_prompt = blip_inf_prompt(real_img)
            ref_prompt = base_prompt.replace("{prompt}", blip_ref_prompt)
            print(ref_prompt)
            inf_prompt = base_prompt.replace("{prompt}", tar_obj)
            print(inf_prompt)
        else:
            #  "the material texture":("{prompt} in a cube, realistic, with a white background, 3d game asset","")

            ref_prompt = blip_inf_prompt(real_img)
            inf_prompt = tar_obj

        for tar_seed in tar_seeds:
            latents.append(init_latent(model=pipe, device_name=device, dtype=torch_dtype, seed=tar_seed,set_height=512, num_inference_steps = num_inference_steps))

        latents = torch.cat(latents, dim=0)

        # with torch.enable_grad():
        with torch.no_grad():

            latents = nn.Parameter(latents)

            print(latents.shape)
            latents0 = latents[:, 0].data.clone()  # 选择每个批量的第一个时间步
            # latents0_squeezed = latents0.squeeze(1)
            print(f"latents0_squeezed.shape: {latents0.shape}")
            latents[0].requires_grad_(False)

            optimizer = torch.optim.SGD([latents], 2)  # 1 or 2
            print(f"latents.requires_grad: {latents.requires_grad}")

            latents_last = latents.data.clone()
            latents_last_e = latents.data.clone()
            initialized_i = -1

            def callback(self, i, t, callback_kwargs):
                global latents_last, latents_last_e, initialized_i
                if initialized_i < i:
                    latents[:, i].data.copy_(callback_kwargs['latents'])
                    latents_last[:, i].copy_(callback_kwargs['latents'])
                    latents_last_e[:, i].copy_(callback_kwargs['latents'])
                    initialized_i = i
                if i < 3:
                    callback_kwargs['latents'] += latents[:, i+1] - latents[:, i+1].detach()
                latents_e = callback_kwargs['latents'].data.clone()
                callback_kwargs['latents'] += latents_last[:, i].detach() - callback_kwargs['latents'].detach()
                callback_kwargs['latents'] += latents_e.detach() - latents_last_e[:, i].detach()
                # callback_kwargs['latents'] += latents[i:(i+1)].detach() - latents_last_e[i:(i+1)].detach()
                callback_kwargs['latents'] += (latents[:, i].detach() - latents_last_e[:, i].detach()) * 0.95796674
                latents_last[:, i].copy_(callback_kwargs['latents'])
                latents_last_e[:, i].data.copy_(latents_e)
                latents[:, i].data.copy_(latents_e)
                return callback_kwargs

        # num_inference_steps = 8
        # with torch.enable_grad():
            images, has_nsfw_concept,image0_second_elements,image_second_elements, image_o_second_elements = pipe(
                prompt=ref_prompt,
                guidance_scale=guidance_scale,
                latents=latents0 + latents[:, 0]-latents[:, 0].detach(),
                num_images_per_prompt=len(tar_seeds),
                target_prompt=inf_prompt,
                num_inference_steps = num_inference_steps,
                use_inf_negative_prompt=False,
                use_advanced_sampling=False,
                use_prompt_as_null=True,
                image=real_img,
                
                return_dict=False,
                output_type = "pt"
            )
        # [real image, fake1, fake2, ... ]
        noub=noub+1

        save_path = os.path.join(result_dir,"test15", "0218{}_{}_{}_.png".format(style_name, tar_obj,noub))
        save_path0 = os.path.join(result_dir,"test15", "0218{}_{}_{}_0.png".format(style_name, tar_obj,noub))
        save_path1 = os.path.join(result_dir,"test15", "0218{}_{}_{}_r.png".format(style_name, tar_obj,noub))
        save_patho = os.path.join(result_dir,"test15", "0218{}_{}_{}_or.png".format(style_name, tar_obj,noub))



        n_row = 1
        n_col = len(tar_seeds)
        grid = create_image_grid(images, n_row, n_col)
        grid.save(save_path)
        print(f"saved to {save_path}")

        n_row = 1
        n_col = len(image0_second_elements)
        grid = create_image_grid(image0_second_elements, n_row, n_col)
        grid.save(save_path0)
        print(f"saved to {save_path0}")

        n_row = 1
        n_col = len(image_second_elements)
        grid = create_image_grid(image_second_elements, n_row, n_col)
        grid.save(save_path1)
        print(f"saved to {save_path1}")

        n_row = 1
        n_col = len(image_o_second_elements)
        grid = create_image_grid(image_o_second_elements, n_row, n_col)
        grid.save(save_patho)
        print(f"saved to {save_patho}")
