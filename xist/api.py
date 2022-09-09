# Load the diffusion model.
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
import torch
import numpy as np
from torchvision.utils import make_grid

from xist.models.diffusion.ddim import DDIMSampler
from xist.models.diffusion.plms import PLMSSampler
from xist.util import instantiate_from_config


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    print("DONE")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(get_device())
    model.eval()
    return model


config = "configs/stable-diffusion/v1-inference.yaml"
config = OmegaConf.load(config)
model = load_model_from_config(config, "/model.ckpt")
device = torch.device(get_device())
model = model.to(device)

# Options
import os

plms = True
outpath = "outputs/txt2img-samples"
outpath = "assets"
sample_path = outpath
# sample_path = os.path.join(outpath, "samples")
batch_size = 1
n_rows = batch_size
n_iter = 1
n_samples = 1
scale = 1.0
H = 512
W = 512
C = 4
f = 8
ddim_steps = 50
ddim_eta = 0.0
start_code = None
skip_save = False
skip_grid = False


if plms:
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)
os.makedirs(outpath, exist_ok=True)

import time
from tqdm import trange, tqdm


def infer(prompt: str):
    print("Running infer!", prompt)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    data = [batch_size * [prompt]]
    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, H // f, W // f]
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(
                        0, 3, 1, 2
                    )

                    if not skip_save:
                        for x_sample in x_checked_image_torch:
                            x_sample = 255.0 * rearrange(
                                x_sample.cpu().numpy(), "c h w -> h w c"
                            )
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

                    if not skip_grid:
                        all_samples.append(x_checked_image_torch)

            if not skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, "n b c h w -> (n b) c h w")
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                img.save(os.path.join(outpath, f"grid-{grid_count:04}.png"))
                grid_count += 1

            toc = time.time()

    print(
        f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy."
    )


# API Stuff
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/")
async def root():
    return {"message": "Hello World"}


from pydantic import BaseModel


class Request(BaseModel):
    prompt: str


from queue import SimpleQueue

requests = SimpleQueue()


@app.post("/infer")
def run_infer(request: Request):
    global requests
    if requests.qsize() > 2:
        return {"message": "Too many requests"}
    prompt = request.prompt
    print("Prompt: ", request)
    requests.put(prompt)
    return {"message": requests.qsize()}


@app.get("/results")
def results():
    files = reversed(sorted(os.listdir("assets")))
    return {"files": list(files)}


import threading


def process_prompts(name):
    print("Starting thread", name)
    while True:
        prompt = requests.get()
        print("Got prompt", prompt)
        infer(prompt)


x = threading.Thread(target=process_prompts, args=(1,))
x.start()
