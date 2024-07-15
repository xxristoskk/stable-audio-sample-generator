import os
import json
import argparse
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str)
parser.add_argument('--steps', type=int) 
parser.add_argument('--seconds', type=int)
parser.add_argument('--scale', type=int)
parser.add_argument('--name', type=str) #name of project
parser.add_argument('--multiprompt', type=bool) #true if using multiple prompts, else false
parser.add_argument('--instrument', type=str) #name of instrument folder
parser.add_argument('--batch', type=int) #number of generations
args = parser.parse_args()

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Additional diagnostics
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CuDNN version:", torch.backends.cudnn.version())
print("CUDA path:", os.getenv("CUDA_PATH"))
print("CUDA visible devices:", os.getenv("CUDA_VISIBLE_DEVICES"))

# Get the hugging face token
with open('keys.json', 'r') as f:
    data = json.load(f)

# huggingface login
login(data['HF_TOKEN'])

# designating the model to run on the GPU or CPU -- not applicable for my purposes
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cuda'

#get the model and define sample rate and size
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# takes in the parameters and runs the model
def run_model(prompt: str, seconds: int, steps: int, output_name: str):
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds
    }]

    output = generate_diffusion_cond(
        model,
        steps = steps,
        cfg_scale = args.scale,
        conditioning = conditioning,
        sample_size = sample_size,
        sigma_min = 0.3,
        sigma_max = 500,
        sampler_type = "dpmpp-3m-sde",
        device = device
    )

    # rearranging the dimensions of the tensor (for the model?)
    output = rearrange(output, "b d n -> d (b n)")

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(output_name, output, sample_rate)

# reads the multiprompt file
def read_txt(path: str):
    with open(path, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    return prompts

# create new directory for output(s)
new_dir = os.path.join(f'{os.getcwd()}/{args.instrument}', args.name)
os.mkdir(new_dir)

# check to see if there is one or multiple prompts, or if there is a batch number
if not args.multiprompt and not args.batch:
    run_model(args.prompt, args.seconds, args.steps, f'{new_dir}/{args.name}.wav')
elif args.batch and not args.multiprompt:
    x = 0
    while x < args.batch:
        run_model(args.prompt, args.seconds, args.steps, f'{new_dir}/{args.name}[{x}].wav')
        x += 1
else:
    prompts = read_txt(f'{args.instrument}/{args.instrument}.txt')
    for x in range(len(prompts)):
        print(f'Prompt: {prompts[x]}')  
        if args.batch:
            i = 0
            while i < args.batch:
                run_model(prompts[x], args.seconds, args.steps, f"{new_dir}/{args.name}[{x}]_{i}.wav")
                i += 1
        else:
            run_model(prompts[x], args.seconds, args.steps, f"{new_dir}/{args.name}[{x}].wav")

print('DONE!')