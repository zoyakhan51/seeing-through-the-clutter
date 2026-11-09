import os

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Gemma4ForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration

from tangle import config
from tangle import data
from tangle.paths import MODELS


def family_of(model):
    """Say whether a loaded model is a qwen or a gemma model"""
    return "gemma" if model.config.model_type.startswith("gemma") else "qwen"


def load(name):
    """Load a downloaded model and its processor the same way study 3 did"""
    spec = config.MODELS[name]
    folder = MODELS/name
    # the cpu tests load a tiny model in full precision through these two variables
    dtype = os.environ.get("TANGLE_DTYPE",config.DTYPE)
    device_map = os.environ.get("TANGLE_DEVICE_MAP",spec["device_map"])
    # gemma picks its own attention code because its audio part cannot use sdpa
    kind = Gemma4ForConditionalGeneration if spec.get("family")=="gemma" else Qwen3VLForConditionalGeneration
    model = kind.from_pretrained(folder,dtype=getattr(torch,dtype),device_map=device_map)
    model.eval()
    kinds = {str(p.dtype) for p in model.parameters()}
    assert kinds=={f"torch.{dtype}"},f"model loaded with weights in {kinds}"
    processor = AutoProcessor.from_pretrained(folder)
    return model,processor


def prompt_text(processor,question,prefill):
    """Build the prompt text exactly like study 3 with the picture before the question and the answer start at the end"""
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    return processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill


def batch_inputs(model,processor,prompt,images):
    """Turn one prompt and a list of pictures into a batch where every row has the same tokens"""
    family = family_of(model)
    # gemma gets the picture already enlarged to 768 px so its processor does not blur pixels across squares
    pictures = [Image.fromarray(data.enlarge(image,family)) for image in images]
    inputs = processor(text=[prompt]*len(pictures),images=pictures,return_tensors="pt",add_special_tokens=False)
    return inputs.to(model.device)


def check_picture(model,inputs):
    """Check that the processor read every picture at the size the squares are built for"""
    if family_of(model)=="qwen":
        assert inputs["image_grid_thw"].tolist()==[[1,2*config.GRID,2*config.GRID]]*len(inputs["image_grid_thw"]),"the processor resized the picture"
        return
    # gemma lists 48 by 48 real patches row by row and then padding patches marked with position -1
    n = config.PROCESSED_SIZE["gemma"]//config.PATCH
    cells = data.patch_cells("gemma")
    real = (inputs["image_position_ids"]!=-1).all(-1)
    assert (real.sum(1)==n*n).all(),"the processor resized the picture"
    assert real[:,:n*n].all(),"the real patches must come first"
    # positions are written as x then y
    expected = torch.as_tensor(cells[:,::-1].copy(),device=inputs["image_position_ids"].device)
    assert (inputs["image_position_ids"][:,:n*n]==expected).all(),"the patches are not listed row by row"


def pixel_patches(model,inputs):
    """Give the pixels of every real patch of every row as rows by patches by numbers"""
    rows = inputs["input_ids"].shape[0]
    if family_of(model)=="qwen":
        return inputs["pixel_values"].reshape(rows,config.PATCHES,-1)
    n = (config.PROCESSED_SIZE["gemma"]//config.PATCH)**2
    return inputs["pixel_values"][:,:n]


def vision_depth(model):
    """Number of blocks in the vision part"""
    if family_of(model)=="qwen":
        return model.config.vision_config.depth
    return model.config.vision_config.num_hidden_layers


def end_token(model,processor):
    """The token that ends the model's turn and so ends every gold answer"""
    if family_of(model)=="qwen":
        return processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    # reading it from the chat template so it is the exact token gemma writes after an answer
    messages = [{"role":"user","content":[{"type":"text","text":"question"}]},{"role":"assistant","content":"answer"}]
    text = processor.apply_chat_template(messages,tokenize=False)
    after = text[text.rindex("answer")+len("answer"):]
    return processor.tokenizer(after,add_special_tokens=False).input_ids[0]


def image_positions(model,input_ids):
    """Find the token positions that hold the picture and check they all sit before the question"""
    ids = input_ids[0]
    positions = torch.nonzero(ids==model.config.image_token_id).flatten()
    assert len(positions)==config.TOKENS,f"expected {config.TOKENS} image tokens and found {len(positions)}"
    assert (input_ids==input_ids[0]).all(),"every row of a batch must have the same tokens"
    assert positions[-1]-positions[0]==config.TOKENS-1,"image tokens must be one block"
    return positions
