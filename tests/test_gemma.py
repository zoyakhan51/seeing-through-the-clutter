import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Gemma4Config
from transformers import Gemma4ForConditionalGeneration
from transformers import Gemma4ImageProcessor

from tangle import config
from tangle import data
from tangle import hooks
from tangle import model as qwen

# the real gemma processor needs the tokenizer from hugging face, so these tests build the inputs
# with gemma's own image processor and a token list laid out like the real prompt
PATCHES = (config.PROCESSED_SIZE["gemma"]//config.PATCH)**2


@pytest.fixture(scope="session")
def image_processor():
    """Gemma's image processor with its default settings, which are the ones the real model uses"""
    return Gemma4ImageProcessor()


@pytest.fixture(scope="session")
def tiny_gemma():
    """A tiny random gemma 4 with the real special token ids, per layer inputs and shared memory layers"""
    text = {"hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
            "global_head_dim":16,"hidden_size_per_layer_input":16,"num_kv_shared_layers":2,"sliding_window":64,
            # the two shared layers at the end reuse the memory of the last unshared layer of their kind
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
            "use_bidirectional_attention":"vision","dtype":"float32"}
    vision = {"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":16,
              "position_embedding_size":64,"dtype":"float32"}
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text,vision_config=vision,audio_config=None)).eval()
    return model


def gemma_inputs(model,image_processor,pictures):
    """Build a batch like the real processor does: the enlarged pictures and the picture tokens between boi and eoi"""
    out = image_processor(images=[Image.fromarray(data.enlarge(p,"gemma")) for p in pictures],return_tensors="pt")
    image = model.config.image_token_id
    ids = [2,105,2364,107,model.config.boi_token_id]+[image]*config.TOKENS+[model.config.eoi_token_id]+list(range(300,360))+[106,107,105,4368,107]
    rows = len(pictures)
    input_ids = torch.tensor([ids]*rows)
    return {
        "input_ids":input_ids,
        "attention_mask":torch.ones_like(input_ids),
        "mm_token_type_ids":(input_ids==image).long(),
        "pixel_values":out["pixel_values"],
        "image_position_ids":out["image_position_ids"],
    }


@pytest.fixture(scope="session")
def gemma_scene():
    """The first objects scene with its clean and cluttered pictures"""
    first = data.load_scenes("objects",1)[0]
    return first,[data.load_image("objects",row) for row in first["rows"]]


def test_enlarge_keeps_squares_whole():
    """Every 32 px square of the png becomes exactly one 48 px block of the enlarged picture"""
    rng = np.random.default_rng(0)
    png = rng.integers(0,256,(512,512,3),dtype=np.uint8)
    big = data.enlarge(png,"gemma")
    assert big.shape==(768,768,3)
    source = np.arange(768)*512//768
    assert (big==png[source][:,source]).all()
    # every pixel of the 48 px block of square s comes from square s of the png
    assert ((source//32)==(np.arange(768)//48)).all()
    # the qwen picture is left as it is
    assert data.enlarge(png,"qwen") is png


def test_processor_keeps_the_enlarged_picture(tiny_gemma,image_processor):
    """A 768 px picture goes in unchanged and its real patches come row by row before the padding"""
    rng = np.random.default_rng(1)
    png = rng.integers(0,256,(512,512,3),dtype=np.uint8)
    inputs = gemma_inputs(tiny_gemma,image_processor,[png])
    qwen.check_picture(tiny_gemma,inputs)
    patches = qwen.pixel_patches(tiny_gemma,inputs)[0]
    big = torch.as_tensor(data.enlarge(png,"gemma")).float()/255
    n = 768//16
    expected = big.reshape(n,16,n,16,3).permute(0,2,1,3,4).reshape(n*n,-1)
    assert torch.allclose(patches,expected,atol=1e-6)


def test_one_pixel_changes_only_its_patches(tiny_gemma,image_processor):
    """Changing one png pixel changes exactly the patches its enlarged copies fall in"""
    rng = np.random.default_rng(2)
    clean = rng.integers(0,256,(512,512,3),dtype=np.uint8)
    cells = data.patch_cells("gemma")
    for _ in range(8):
        y,x = rng.integers(0,512,2)
        changed = clean.copy()
        changed[y,x] = 255-changed[y,x]
        inputs = gemma_inputs(tiny_gemma,image_processor,[clean,changed])
        pixels = qwen.pixel_patches(tiny_gemma,inputs)
        differ = set(torch.nonzero((pixels[0]!=pixels[1]).any(-1)).flatten().tolist())
        source = np.arange(768)*512//768
        rows = set(np.flatnonzero(source==y)//16)
        cols = set(np.flatnonzero(source==x)//16)
        expected = {i for i,(r,c) in enumerate(cells) if r in rows and c in cols}
        assert differ==expected


def test_masks_match_processed_pixels(tiny_gemma,image_processor):
    """Untouched patches from the png must be exactly the patches gemma's processor leaves unchanged"""
    squares = data.patch_squares("gemma")
    for first in data.load_scenes("objects",3):
        clean = data.load_image("objects",first["rows"][0])
        for row in first["rows"][1:]:
            image = data.load_image("objects",row)
            m = data.masks(clean,image,data.objects_of(row),"gemma")
            pixels = qwen.pixel_patches(tiny_gemma,gemma_inputs(tiny_gemma,image_processor,[clean,image]))
            same = (pixels[0]==pixels[1]).all(-1).numpy()
            assert (same==m["untouched_patch"]).all()
            assert m["object"].sum()>0 and m["clutter"].sum()>0
            assert not (m["object"]&m["clutter"]).any()
            # a square is untouched only when all nine of its patches are
            whole = np.ones(config.TOKENS,bool)
            np.logical_and.at(whole,squares,m["untouched_patch"])
            assert (whole==m["untouched"]).all()
            # a patch on an object lies in a square that is on the object or was changed by clutter
            assert (m["object"]|m["clutter"])[squares[m["object_patch"]]].all()


def test_qwen_patch_boxes_are_unchanged():
    """The new box rule gives qwen exactly the patches the old rule gave"""
    rng = np.random.default_rng(3)
    for _ in range(50):
        x0,y0 = rng.integers(0,500,2)
        x1,y1 = x0+rng.integers(0,60),y0+rng.integers(0,60)
        assert (data.bbox_patches((x0,y0,x1,y1),"qwen")==data.bbox_cells((x0,y0,x1,y1),config.PATCH)).all()


def test_recorder_is_exact(tiny_gemma,image_processor,gemma_scene):
    """Two copies of the clean picture give identical notes and untouched patches start out identical"""
    model = tiny_gemma
    first,pictures = gemma_scene
    recs = []
    for picture in [pictures[0],pictures[0],pictures[2]]:
        inputs = gemma_inputs(model,image_processor,[picture])
        positions = qwen.image_positions(model,inputs["input_ids"])
        with hooks.Recorder(model,positions,1) as rec,torch.no_grad():
            model.model(**inputs)
        recs.append(rec)
    clean,again,cluttered = recs
    assert len(clean.vision)==qwen.vision_depth(model)+1
    assert len(clean.llm)==model.config.text_config.num_hidden_layers+1
    assert clean.deepstack==[]
    assert clean.vision[0].shape[1]==PATCHES
    for a,b in zip(clean.vision+clean.llm,again.vision+again.llm):
        assert torch.equal(a,b)
    m = data.masks(pictures[0],pictures[2],data.objects_of(first["rows"][2]),"gemma")
    untouched = torch.as_tensor(m["untouched_patch"])
    assert torch.equal(cluttered.vision[0][0][untouched],clean.vision[0][0][untouched])
    assert not torch.equal(cluttered.vision[0][0][~untouched],clean.vision[0][0][~untouched])
    # attention spreads the clutter so untouched squares change later on
    squares = torch.as_tensor(m["untouched"])
    assert not torch.equal(cluttered.llm[-1][0][squares],clean.llm[-1][0][squares])


def side_inputs(model,positions):
    """Keep the extra per layer input of the picture tokens as it enters every decoder layer"""
    kept = []

    def hook(module,args,kwargs):
        side = args[1] if len(args)>1 else kwargs.get("per_layer_input")
        if side.shape[1]>1:
            kept.append(side[:,positions].clone())
    handles = [layer.register_forward_pre_hook(hook,with_kwargs=True) for layer in model.model.language_model.layers]
    return kept,handles


def test_swap_rows_take_the_right_notes(tiny_gemma,image_processor,gemma_scene):
    """Every swap row gets clean or cluttered notes and side inputs exactly where its plan says"""
    model = tiny_gemma
    first,pictures = gemma_scene
    row = first["rows"][2]
    m = data.masks(pictures[0],pictures[2],data.objects_of(row),"gemma")
    parts = {"object":m["object"],"background":m["background"],"clutter":m["clutter"]}
    names = ["clean","cluttered","A","B","C","D","B_obj","B_bg","B_noise","B_hand","C_hand","cut_2"]
    plans = [None,None]+[config.SWAP[n] for n in names[2:]]
    inputs = gemma_inputs(model,image_processor,[pictures[0]]+[pictures[2]]*(len(names)-1))
    positions = qwen.image_positions(model,inputs["input_ids"])
    with hooks.Surgery(model,positions,plans,parts,seed=7),hooks.Recorder(model,positions,len(names)) as rec,torch.no_grad():
        # the side input hooks go on after the swap's own hooks so they see what the swap hands each layer
        kept,handles = side_inputs(model,positions)
        logits = model(**inputs,logits_to_keep=4).logits
    for handle in handles:
        handle.remove()
    r = {n:i for i,n in enumerate(names)}
    obj = torch.as_tensor(m["object"])
    bg = torch.as_tensor(m["background"])
    clut = torch.as_tensor(m["clutter"])
    untouched = obj|bg
    layers = model.config.text_config.num_hidden_layers
    assert len(kept)==layers
    for k in range(layers):
        for h in [rec.llm[k],kept[k]]:
            assert torch.equal(h[r["A"]],h[1])
            assert torch.equal(h[r["D"]],h[0])
            assert torch.equal(h[r["B"]][untouched],h[0][untouched]) and torch.equal(h[r["B"]][clut],h[1][clut])
            assert torch.equal(h[r["C"]][untouched],h[1][untouched]) and torch.equal(h[r["C"]][clut],h[0][clut])
            assert torch.equal(h[r["B_obj"]][obj],h[0][obj]) and torch.equal(h[r["B_obj"]][bg|clut],h[1][bg|clut])
            assert torch.equal(h[r["B_bg"]][bg],h[0][bg]) and torch.equal(h[r["B_bg"]][obj|clut],h[1][obj|clut])
            moved = (h[r["B_noise"]][untouched]-h[1][untouched]).norm(dim=-1)
            leak = (h[0][untouched]-h[1][untouched]).norm(dim=-1)
            assert torch.allclose(moved,leak,rtol=1e-4,atol=1e-5) and torch.equal(h[r["B_noise"]][clut],h[1][clut])
    # the clean and cluttered side inputs really differ so the side input test means something
    assert not torch.equal(kept[0][0][untouched],kept[0][1][untouched])
    # the hand over rows change the projected squares and gemma then builds their side inputs from them
    hand = rec.llm[0]
    assert torch.equal(hand[r["B_hand"]][untouched],hand[0][untouched]) and torch.equal(hand[r["B_hand"]][clut],hand[1][clut])
    assert torch.equal(hand[r["C_hand"]][untouched],hand[1][untouched]) and torch.equal(hand[r["C_hand"]][clut],hand[0][clut])
    assert torch.allclose(kept[0][r["B_hand"]][untouched],kept[0][0][untouched],atol=1e-6)
    assert torch.allclose(kept[0][r["B_hand"]][clut],kept[0][1][clut],atol=1e-6)
    # a row cut at layer 2 is plain cluttered before it, notes and side input alike
    for k in [0,1]:
        assert torch.equal(rec.llm[k][r["cut_2"]],rec.llm[k][1]) and torch.equal(kept[k][r["cut_2"]],kept[k][1])
    assert torch.equal(rec.llm[2][r["cut_2"]][untouched],rec.llm[2][0][untouched])
    assert torch.equal(kept[2][r["cut_2"]][untouched],kept[2][0][untouched])
    assert torch.equal(logits[r["A"]],logits[1])
    assert torch.equal(logits[r["D"]],logits[0])
    assert not torch.equal(logits[r["B"]],logits[r["A"]])


def test_generation_keeps_identities(tiny_gemma,image_processor,gemma_scene):
    """Greedy answers of row A equal the plain cluttered row and row D the plain clean row"""
    model = tiny_gemma
    first,pictures = gemma_scene
    row = first["rows"][3]
    m = data.masks(pictures[0],pictures[3],data.objects_of(row),"gemma")
    parts = {"object":m["object"],"background":m["background"],"clutter":m["clutter"]}
    names = ["clean","cluttered","A","D","B"]
    plans = [None,None]+[config.SWAP[n] for n in names[2:]]
    inputs = gemma_inputs(model,image_processor,[pictures[0]]+[pictures[3]]*4)
    positions = qwen.image_positions(model,inputs["input_ids"])
    with hooks.Surgery(model,positions,plans,parts,seed=1),torch.no_grad():
        out = model.generate(**inputs,max_new_tokens=12,do_sample=False)
    assert torch.equal(out[2],out[1])
    assert torch.equal(out[3],out[0])
    with torch.no_grad():
        alone = model.generate(**gemma_inputs(model,image_processor,[pictures[3]]),max_new_tokens=12,do_sample=False)
    assert torch.equal(alone[0],out[1])
