import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import normalize
from torchvision.models import resnet50, ResNet50_Weights
from src.distance import group_percent_explained
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def modify_text(orig_prompt, features, features_diff):
    prompt = orig_prompt
    for nonzero in np.nonzero(features_diff)[0]:
        if features_diff[nonzero] < 0:
            prompt = prompt.replace(str(features[nonzero]), "", int(-1 * features_diff[nonzero]))
        else:
            prompt = ", ".join([str(features[nonzero]).upper()] * int(features_diff[nonzero])) + ", " + prompt
    return prompt


def text2img(prompts):
    model_id = "CompVis/stable-diffusion-v1-4"

    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda:1")

    num_inference_steps = 50
    pipe.scheduler.set_timesteps(num_inference_steps, device=torch.device("cuda:1"))

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

def counterfactual_img2img(orig_prompt, features, features_diff):
    cf_prompts = []
    for i, prompt in enumerate(orig_prompt):
        cf_prompts.append(modify_text(prompt, features, features_diff[i, :]))
    counterfactual_imgs = text2img(cf_prompts)
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