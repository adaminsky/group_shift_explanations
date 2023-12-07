import re
import torch
import pickle
from clip_interrogator import Config, Interrogator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision.transforms import Resize
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Resize
from torch.nn.functional import normalize
from src.distance import W2_dist
from torchvision.models import resnet50, ResNet50_Weights
from src.distance import group_percent_explained
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import timm
from tqdm import tqdm


def modify_text(orig_prompt, features, features_diff):
    prompt = orig_prompt
    for nonzero in np.nonzero(features_diff)[0]:
        if features_diff[nonzero] < 0:
            prompt = prompt.replace(str(features[nonzero]), "", int(-1 * features_diff[nonzero]))
        else:
            prompt = ", ".join([str(features[nonzero]).upper()] * int(features_diff[nonzero])) + ", " + prompt
    return prompt


def img2text(imgs):
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", device="cuda"))
    captions = []
    for img in tqdm(imgs):
        caption = ci.interrogate_fast(img)
        captions.append(caption)
    return captions


def text2img(prompts, finetuned=False):
    model_id = "CompVis/stable-diffusion-v1-4"
    finetuned_model_id = "scripts/breeds-model-lora"

    # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    if finetuned:
        pipe.unet.load_attn_procs(finetuned_model_id)
    pipe = pipe.to("cuda")

    num_inference_steps = 50
    pipe.scheduler.set_timesteps(num_inference_steps, device=torch.device("cuda"))

    counterfactual_images = []
    for i, prompt in enumerate(prompts):
        gen = pipe(prompt=prompt,
            width=512,
            height=512,
            # latents=latents[-1],
            guidance_scale=7.5,
            # init_image=Image.fromarray(img_s),
            # strength=0.5,
        ).images[0]
        counterfactual_images.append(gen)
    return counterfactual_images

def counterfactual_img2img(orig_prompt, features, features_diff, finetuned=False):
    cf_prompts = []
    for i, prompt in enumerate(orig_prompt):
        cf_prompts.append(modify_text(prompt, features, features_diff[i, :]))
    counterfactual_imgs = text2img(cf_prompts, finetuned=finetuned)
    return counterfactual_imgs, cf_prompts


def clip_embed(images):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs_source = processor(images=images, return_tensors="pt", padding=True)
    source_emb = normalize(model.get_image_features(**inputs_source)).detach().numpy()

    return source_emb

def resnet50_embed(images):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to("cuda")
    model.eval()
    model.fc = torch.nn.Identity()

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    source_emb = []
    target_emb = []
    cf_emb = []
    for img in images:
        source_pre = preprocess(torch.from_numpy(np.array(img)).permute((2, 0, 1))).cuda()

        source_emb.append(model(source_pre[None, :]).detach().cpu().numpy())
    source_emb = np.concatenate(source_emb, axis=0)

    return source_emb

def resnet50_classify(images):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to("cuda")
    model.eval()

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    source_cls = []
    for img in images:
        source_pre = preprocess(torch.from_numpy(np.array(img)).permute((2, 0, 1))).cuda()
        source_cls.append(model(source_pre[None, :]).detach().cpu().numpy())
    source_cls = np.argmax(np.concatenate(source_cls, axis=0), axis=1)

    return source_cls


def fasterrcnn_embed(imgs):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to("cuda")
    model.eval()
    model.fc = torch.nn.Identity()

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    preprocess = weights.transforms()

    source_emb = []
    for src_img in imgs:
        source_pre = preprocess(torch.from_numpy(np.array(src_img)).permute((2, 0, 1))).cuda()

        source_emb.append(model.backbone(source_pre[None, :])['pool'].reshape((1, -1)).detach().cpu().numpy())

    source_emb = np.concatenate(source_emb, axis=0)
    return source_emb


def img2img_counterfactual_exp(sample, features, source, source_t, source_t_group, target, source_captions, source_groups, target_groups, source_imgs, target_imgs):
    diff_group = (source_t_group - source)[sample].round(0)
    diff_tot = (source_t - source)[sample].round(0)

    counterfactual_images, ot_prompts = counterfactual_img2img([source_captions[i] for i in sample], features, diff_tot)
    counterfactual_images_group, got_prompts = counterfactual_img2img([source_captions[i] for i in sample], features, diff_group)

    diff = np.nonzero(np.any(diff_group != diff_tot, axis=1))[0]
    print("Differences:", diff.shape)

    # Get distance in CLIP embedding space
    source_emb = clip_embed(source_imgs)
    target_emb = clip_embed(target_imgs)
    cf_emb = clip_embed(counterfactual_images)
    cf_g_emb = clip_embed(counterfactual_images_group)


    no_group_s = np.ones((source_emb.shape[0], 1))
    no_group_t = np.ones((target_emb.shape[0], 1))
    group_percent_explained(source_emb, cf_emb, target_emb[sample], source_groups[sample], target_groups[sample], ["g1", "g2", "g3", "g4"])
    group_percent_explained(source_emb, cf_g_emb, target_emb[sample], source_groups[sample], target_groups[sample], ["g1", "g2", "g3", "g4"])

    # Get distance in ResNet50 embedding space

    source_emb = resnet50_embed(source_imgs)
    target_emb = resnet50_embed(target_imgs)
    cf_emb = resnet50_embed(counterfactual_images)
    cf_g_emb = resnet50_embed(counterfactual_images_group)


    no_group_s = np.ones((source_emb.shape[0], 1))
    no_group_t = np.ones((target_emb.shape[0], 1))
    group_percent_explained(source_emb, cf_emb, target_emb[sample], source_groups[sample], target_groups[sample], ["g1", "g2", "g3", "g4"])
    group_percent_explained(source_emb, cf_g_emb, target_emb[sample], source_groups[sample], target_groups[sample], ["g1", "g2", "g3", "g4"])


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, x0 ],dim = 0)

    return xts


def get_variance(model, timestep): #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                    model_output,
                                    torch.LongTensor([next_timestep]))
    return next_sample

def inversion_forward_process(model, x0,
                            etas = None,
                            prog_bar = False,
                            prompt = "",
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None):

    if not prompt=="":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xt = x0
    op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)

    for t in op:
        idx = t_to_idx[int(t)]
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]

        with torch.no_grad():
            out = model.unet.forward(xt.to(torch.float16), timestep =  t, encoder_hidden_states = uncond_embedding.to(torch.float16))
            if not prompt=="":
                cond_out = model.unet.forward(xt.to(torch.float16), timestep=t, encoder_hidden_states = text_embeddings.to(torch.float16))

        if not prompt=="":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 =  xts[idx+1][None]
            # pred of x0
            pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )*z
            xts[idx+1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z =  eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(model,
                    xT,
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None,
                    controller=None,
                    asyrp = False):

    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = t_to_idx[int(t)]
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(xt.to(torch.float16), timestep =  t,
                                            encoder_hidden_states = uncond_embedding.to(torch.float16))

            ## Conditional embedding
        if prompts:
            with torch.no_grad():
                cond_out = model.unet.forward(xt.to(torch.float16), timestep =  t,
                                                encoder_hidden_states = text_embeddings.to(torch.float16))


        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else:
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z)
        if controller is not None:
            xt = controller.step_callback(xt)
    return xt, zs

def ddim_invert(sd_pipe, x0:torch.FloatTensor, prompt_src:str ="", num_inference_steps=100, cfg_scale_src = 3.5, eta = 1, num_diffusion_steps=50):

  #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
  #  based on the code in https://github.com/inbarhub/DDPM_inversion

  #  returns wt, zs, wts:
  #  wt - inverted latent
  #  wts - intermediate inverted latents
  #  zs - noise maps

  sd_pipe.scheduler.set_timesteps(num_diffusion_steps)

  # vae encode image
  with torch.autocast("cuda"), torch.inference_mode():
      w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).to(torch.float16)

  # find Zs and wts - forward process
  wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=num_diffusion_steps)
  return zs, wts


def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = torchvision.transforms.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def add_margin(pil_img, top = 0, right = 0, bottom = 0,
                    left = 0, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)

    result.paste(pil_img, (left, top))
    return result


def image_grid(imgs, rows = 1, cols = None,
                    size = None,
                   titles = None, text_pos = (0, 0)):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)

    if not size is None:
        imgs = [img.resize((size,size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows*cols

    top=20
    w, h = imgs[0].size
    delta = 0
    if len(imgs)> 1 and not imgs[1].size[1] == h:
        delta = top
        h = imgs[1].size[1]
    if not titles is  None:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                                    size = 20, encoding="unic")
        h = top + h
    grid = Image.new('RGB', size=(cols*w, rows*h+delta))
    for i, img in enumerate(imgs):

        if not titles is  None:
            img = add_margin(img, top = top, bottom = 0,left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i],(0,0,0),
            font = font)
        if not delta == 0 and i > 0:
           grid.paste(img, box=(i%cols*w, i//cols*h+delta))
        else:
            grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid


def sample(sd_pipe, zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):

    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process(sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[skip:])

    # vae decode image
    with torch.autocast("cuda"), torch.inference_mode():
        x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim()<4:
        x0_dec = x0_dec[None,:,:,:]
    img = image_grid(x0_dec)
    return img


def ddim_cf_generate(source_imgs, orig_prompt, features, features_diff):
    cf_prompts = []
    for i, prompt in enumerate(orig_prompt):
        cf_prompts.append(modify_text(prompt, features, features_diff[i, :]))
    print(cf_prompts)

    sd_model_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16).to(device)
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")

    cf_imgs = []
    for img, cf_prompt in zip(source_imgs, cf_prompts):
        image = torch.from_numpy(img).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)

        zs, wts = ddim_invert(sd_pipe, x0=image , prompt_src="", num_inference_steps=50, cfg_scale_src=3.5)
        ddpm_out_img = sample(sd_pipe, zs, wts, prompt_tar=cf_prompt, skip=36, cfg_scale_tar=20)
        cf_imgs.append(ddpm_out_img)
    return cf_imgs, cf_prompts