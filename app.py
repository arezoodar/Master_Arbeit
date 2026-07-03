from flask import Flask, request, jsonify, render_template

from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
import anthropic
import PIL.Image
import json
import re
import io
import base64
import time
import torch

app = Flask(__name__)

groq_client = Groq(api_key="dein-api-key")
openai_client = OpenAI(api_key="dein-api-key")
claude_client = anthropic.Anthropic(api_key="dein-api-key")

print("Lade CLIP Modell...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP geladen!")

PROMPT = """Analysiere diesen Gegenstand und antworte NUR mit JSON, keine Codeblöcke:
{
  "objekt": "Name des Gegenstands",
  "material": "Material(ien)",
  "zustand": "gut/mittel/schlecht",
  "recycelbar": true/false,
  "empfehlung": "Empfehlung: z.B. Weiterverkaufen, Spenden, Reparieren, Recyceln oder Entsorgen"
}"""

categories = {
    "kitchen": [
        "a photo of a pot for cooking",
        "a photo of a plate for eating or plate",
        "a photo of a pressure cooker",
        "a photo of a glass for drinking",
        "a photo of an electric kettle",
        "a photo of a rice cooker",
        "a photo of a hand mixer",
        "a photo of a thermos or insulated flask or thermos",
        "a photo of a glas baking dish or glas dish",
        "a photo of a measuring cup",
        "a photo of a crystal glas",
        "a photo of a whisk",
    ],
    "furniture": [
        "a photo of an armchair ",
        "a photo of a  chair ",
        "a photo of a  sideboard or buffet",
        "a photo of a  wardrobe ",
        "a photo of a  chest ",
        "a photo of a  sofa ",
        "a photo of a  cabinet ",
        "a photo of a  clothes wardrobe",
        "a photo of a  shoe rack or shoe cabinet",
        "a photo of a  console table",
        "a photo of a  drawers",
        "a photo of a  stool ",
    ],
    "home_decor": [
        "a photo of a painting easel",
        "a photo of a painting or artwork on a canvas",
        "a photo of a pen or ballpoint pen",
        "a photo of a vase",
        "a photo of a table lamp or lampshade",
        "a photo of a chandelier",
        "a photo of a table clock",
        "a photo of a flower or plant",
        "a photo of a hookah or shisha pipe or water pipe",
        "a photo of a mirror",
        "a photo of a plant pot or flower pot",
        "a photo of a shelf",
        "a photo of a lamp or hanging lamp or pendant light",
    ],
    "clothing": [
        "a photo of boots or ankle boots",
        "a photo of a cocktail dress",
        "a photo of a blouse or womens shirt",
        "a photo of pants or trousers",
        "a photo of a jacket or blazer",
        "a photo of a coat or overcoat",
        "a photo of childrens clothing",
        "a photo of socks",
        "a photo of a cardigan or knitwear",
        "a photo of a t-shirt",
        "a photo of a hoodie or sweatshirt",
        "a photo of a long coat",
        "a photo of a jeans",
        "a photo of a tunic",
        "a photo of a suit jacket",
    ],
    "electronics": [
        "a photo of a mobile phone or smartphone",
        "a photo of a cordless phone",
        "a photo of a TV or television screen",
        "a photo of a washing machine",
        "a photo of a laptop computer",
        "a photo of a hair dryer",
        "a photo of a dishwasher",
        "a photo of a hairdryer or blow dryer",
        "a photo of a vacuum or vacuum cleaner",
    ],
}

category_prompts = [
    "a photo of kitchen utensils or appliances",
    "a photo of furniture for a living room or bedroom",
    "a photo of home decor items like vases lamps or mirrors",
    "a photo of clothing or fashion items",
    "a photo of electronic devices or appliances",
]
category_keys = list(categories.keys())

zustand_options = [
    "a photo of an object in good condition",
    "a photo of an object in medium or average condition",
    "a photo of an object in bad or damaged condition",
]

def clip_classify(image, options):
    inputs = clip_processor(text=options, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = probs.argmax().item()
    best = options[best_idx]
    confidence = round(probs.max().item() * 100, 1)
    clean = (best
             .replace("a photo of an object made of ", "")
             .replace("a photo of an object in ", "")
             .replace("a photo of an ", "")
             .replace("a photo of a ", "")
             .replace("a photo of ", ""))
    return clean, confidence

def clip_classify_hierarchical(image_bytes):
    image = PIL.Image.open(io.BytesIO(image_bytes))

    all_prompts = []
    all_cat_keys = []
    for cat_key, prompts in categories.items():
        for p in prompts:
            all_prompts.append(p)
            all_cat_keys.append(cat_key)

    inputs = clip_processor(
        text=all_prompts, images=image,
        return_tensors="pt", padding=True
    )
    probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
    best_idx = probs.argmax().item()
    best_conf = round(probs[best_idx].item() * 100, 1)

    obj_name = (all_prompts[best_idx]
                .replace("a photo of an ", "")
                .replace("a photo of a ", "")
                .replace("a photo of ", "")
                .strip())

    return {
        "category":      all_cat_keys[best_idx],
        "category_conf": best_conf,
        "object":        obj_name,
        "object_conf":   best_conf,
    }

def analyze_llama(image_bytes):
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": PROMPT}
        ]}]
    )
    text = re.sub(r'```json|```', '', response.choices[0].message.content).strip()
    return json.loads(text)

def analyze_gpt4o(image_bytes):
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": PROMPT}
        ]}])
    text = re.sub(r'```json|```', '', response.choices[0].message.content).strip()
    return json.loads(text)

def analyze_claude(image_bytes):
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    response = claude_client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": PROMPT}
        ]}])
    text = re.sub(r'```json|```', '', response.content[0].text).strip()
    return json.loads(text)

def analyze_clip(image_bytes):
    image = PIL.Image.open(io.BytesIO(image_bytes))
    hier = clip_classify_hierarchical(image_bytes)

    obj_name = hier['object'].split(" or ")[0]

    # ── Material erkennen ──
    material_prompts = [
        f"a photo of {obj_name} made of wood",
        f"a photo of {obj_name} made of metal",
        f"a photo of {obj_name} made of plastic",
        f"a photo of {obj_name} made of textile or fabric",
        f"a photo of {obj_name} made of glass",
        f"a photo of {obj_name} made of ceramic",
        f"a photo of {obj_name} made of leather",
    ]
    material_keys_local = [
        "wood", "metal", "plastic", "textile or fabric",
        "glass", "ceramic", "leather"
    ]

    mat_inputs = clip_processor(
        text=material_prompts, images=image,
        return_tensors="pt", padding=True
    )
    mat_probs = clip_model(**mat_inputs).logits_per_image.softmax(dim=1)
    mat_idx = mat_probs[0].argmax().item()
    mat_key = material_keys_local[mat_idx]
    mat_conf = round(mat_probs[0].max().item() * 100, 1)

    # ── Zustand erkennen ──
    zus, zus_conf = clip_classify(image, zustand_options)
    zustand_map = {
        "good condition":              "gut",
        "medium or average condition": "mittel",
        "bad or damaged condition":    "schlecht"
    }
    zustand = zustand_map.get(zus, "mittel")

    # ── Recycelbar — basiert auf Material ──
    recycling_rules_updated = {
        "wood":              True,
        "metal":             True,
        "glass":             True,
        "leather":           True,
        "plastic":           True,
        "textile or fabric": True,
        "ceramic":           False,
    }
    recycelbar = recycling_rules_updated.get(mat_key, False)

    # Empfehlung — basiert auf Zustand + Recycelbar 
    if zustand == "gut":
        empfehlung = "Weiterverkaufen"
    elif zustand == "mittel":
        empfehlung = "Weiterverkaufen oder Spenden"
    else:  # schlecht
        if recycelbar:
            empfehlung = "Recyceln"
        else:
            empfehlung = "Entsorgen"

    return {
        "objekt":     f"{hier['object']} ({hier['object_conf']}%)",
        "material":   f"{mat_key} ({mat_conf}%)",
        "zustand":    zustand,
        "recycelbar": recycelbar,
        "empfehlung": empfehlung,
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    model = request.form.get('model', 'llama')
    image_bytes = file.read()
    start_time = time.time()

    if model == 'llama': result = analyze_llama(image_bytes)
    elif model == 'gpt4o': result = analyze_gpt4o(image_bytes)
    elif model == 'claude': result = analyze_claude(image_bytes)
    elif model == 'clip': result = analyze_clip(image_bytes)

    result['zeit'] = round(time.time() - start_time, 2)
    return jsonify(result)

@app.route('/analyze_all', methods=['POST'])
def analyze_all():
    file = request.files['image']
    image_bytes = file.read()
    results = {}

    for model_name, func in [
        ('gpt4o', analyze_gpt4o),
        ('claude', analyze_claude),
        ('clip', analyze_clip),
        ('llama', analyze_llama),
    ]:
        start_time = time.time()
        try:
            result = func(image_bytes)
            result['zeit'] = round(time.time() - start_time, 2)
            results[model_name] = result
        except Exception as e:
            print(f"ERROR in {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
