"""
Evaluation Script - Masterarbeit KI-basierte Bildanalyse
---------------------------------------------------------
Liest alle Bilder aus dem Picture-Ordner,
schickt jedes Bild an alle 4 Modelle,
und speichert die Ergebnisse in einer CSV-Datei.

Verwendung:
    python evaluation_script.py
"""

import os
import csv
import json
import re
import io
import base64
import time

from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from transformers import CLIPProcessor, CLIPModel
import PIL.Image
import torch

# ============================================================
# API KEYS - hier eintragen
# ============================================================
OPENAI_API_KEY  = "dein-api-key"
ANTHROPIC_KEY   = "dein-api-key"
GROQ_API_KEY    = "dein-api-key"

# ============================================================
# PFAD ZUM PICTURE-ORDNER (Desktop)
# ============================================================
import platform
if platform.system() == "Darwin":  # macOS
    PICTURE_FOLDER = os.path.expanduser("~/Desktop/Picture")
elif platform.system() == "Windows":
    PICTURE_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "Picture")
else:
    PICTURE_FOLDER = os.path.expanduser("~/Desktop/Picture")

# Ausgabe-CSV
OUTPUT_CSV = os.path.join(os.path.dirname(PICTURE_FOLDER), "evaluation_results.csv")

# ============================================================
# CLIENTS
# ============================================================
openai_client  = OpenAI(api_key=OPENAI_API_KEY)
claude_client  = Anthropic(api_key=ANTHROPIC_KEY)
groq_client    = Groq(api_key=GROQ_API_KEY)

print("Lade CLIP Modell...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP geladen!")

# ============================================================
# PROMPT
# ============================================================
PROMPT = """Analysiere diesen Gegenstand und antworte NUR mit JSON, keine Codeblöcke:
{
  "objekt": "Name des Gegenstands",
  "material": "Material(ien)",
  "zustand": "gut/mittel/schlecht",
  "recycelbar": true/false,
  "empfehlung": "Empfehlung: z.B. Weiterverkaufen, Spenden, Reparieren, Recyceln oder Entsorgen"
}"""

# ============================================================
# CLIP KONFIGURATION
# ============================================================
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

material_keys = [
    "wood", "metal", "plastic", "textile or fabric",
    "glass", "leather", "rubber", "ceramic", "cardboard", "foam"
]

zustand_options = [
    "a photo of an object in good condition",
    "a photo of an object in medium or average condition",
    "a photo of an object in bad or damaged condition",
]

recycling_rules = {
    "wood": True, "metal": True, "glass": True, "cardboard": True,
    "plastic": False, "textile or fabric": False, "leather": False,
    "rubber": False, "ceramic": False, "foam": False
}
empfehlung_rules = {
    "wood": "Wertstoffhof", "metal": "Wertstoffhof (Metall)",
    "glass": "Glascontainer", "cardboard": "Papiertonne",
    "plastic": "Sperrmüll", "textile or fabric": "Altkleidercontainer",
    "leather": "Sperrmüll", "rubber": "Sperrmüll",
    "ceramic": "Restmüll", "foam": "Sperrmüll"
}

# ============================================================
# MODELL FUNKTIONEN
# ============================================================
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

def analyze_llama(image_bytes):
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": PROMPT}
        ]}])
    text = re.sub(r'```json|```', '', response.choices[0].message.content).strip()
    return json.loads(text)

def analyze_clip(image_bytes):
    image = PIL.Image.open(io.BytesIO(image_bytes))

    # Step 1+2: Flat classification - alle Objekte auf einmal vergleichen
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
    obj_conf = round(probs[best_idx].item() * 100, 1)
    cat_conf = obj_conf

    obj_name = (all_prompts[best_idx]
                .replace("a photo of an ", "")
                .replace("a photo of a ", "")
                .replace("a photo of ", "")
                .strip())

    # Step 3: Material
    material_prompts = [
        f"a photo of {obj_name.split(' or ')[0]} made of {m}" for m in material_keys
    ]
    mat_inputs = clip_processor(
        text=material_prompts, images=image,
        return_tensors="pt", padding=True
    )
    mat_probs = clip_model(**mat_inputs).logits_per_image.softmax(dim=1)
    mat_idx = mat_probs.argmax().item()
    mat_key = material_keys[mat_idx]
    mat_conf = round(mat_probs.max().item() * 100, 1)

    # Step 4: Zustand
    zus_inputs = clip_processor(
        text=zustand_options, images=image,
        return_tensors="pt", padding=True
    )
    zus_probs = clip_model(**zus_inputs).logits_per_image.softmax(dim=1)[0]
    zus_idx = zus_probs.argmax().item()
    zustand_map = {
        0: "gut",
        1: "mittel",
        2: "schlecht"
    }

    return {
        "objekt":     f"{obj_name} ({cat_conf}% / {obj_conf}%)",
        "material":   f"{mat_key} ({mat_conf}%)",
        "zustand":    zustand_map[zus_idx],
        "recycelbar": recycling_rules.get(mat_key, False),
        "empfehlung": empfehlung_rules.get(mat_key, "Sperrmüll"),
    }

# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    # CSV Header
    fieldnames = [
        "image_id", "kategorie", "dateipfad",
        # GPT-4o
        "gpt4o_objekt", "gpt4o_material", "gpt4o_zustand",
        "gpt4o_recycelbar", "gpt4o_empfehlung", "gpt4o_zeit",
        # Claude
        "claude_objekt", "claude_material", "claude_zustand",
        "claude_recycelbar", "claude_empfehlung", "claude_zeit",
        # Llama
        "llama_objekt", "llama_material", "llama_zustand",
        "llama_recycelbar", "llama_empfehlung", "llama_zeit",
        # CLIP
        "clip_objekt", "clip_material", "clip_zustand",
        "clip_recycelbar", "clip_empfehlung", "clip_zeit",
    ]

    # Bestehende Ergebnisse laden (falls Script unterbrochen wurde)
    processed = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["image_id"])
        print(f"📂 {len(processed)} bereits verarbeitete Bilder gefunden.")

    # CSV öffnen
    write_header = not os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # Bilder durchgehen
    total = 0
    success = 0

    for kategorie in sorted(os.listdir(PICTURE_FOLDER)):
        kat_path = os.path.join(PICTURE_FOLDER, kategorie)
        if not os.path.isdir(kat_path):
            continue

        for filename in sorted(os.listdir(kat_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_id = f"{kategorie}_{filename}"
            total += 1

            # Bereits verarbeitet?
            if image_id in processed:
                print(f"⏭️  Übersprungen: {image_id}")
                continue

            filepath = os.path.join(kat_path, filename)
            print(f"\n🖼️  Verarbeite: {image_id}")

            with open(filepath, "rb") as f:
                image_bytes = f.read()

            row = {
                "image_id": image_id,
                "kategorie": kategorie,
                "dateipfad": filepath,
            }

            # GPT-4o
            print("  → GPT-4o...", end=" ")
            try:
                start = time.time()
                r = analyze_gpt4o(image_bytes)
                row["gpt4o_objekt"]     = r.get("objekt", "")
                row["gpt4o_material"]   = r.get("material", "")
                row["gpt4o_zustand"]    = r.get("zustand", "")
                row["gpt4o_recycelbar"] = r.get("recycelbar", "")
                row["gpt4o_empfehlung"] = r.get("empfehlung", "")
                row["gpt4o_zeit"]       = round(time.time() - start, 2)
                print(f"✅ ({row['gpt4o_zeit']}s)")
            except Exception as e:
                print(f"❌ {e}")
                row.update({k: "ERROR" for k in fieldnames if k.startswith("gpt4o_")})

            # Claude
            print("  → Claude...", end=" ")
            try:
                start = time.time()
                r = analyze_claude(image_bytes)
                row["claude_objekt"]     = r.get("objekt", "")
                row["claude_material"]   = r.get("material", "")
                row["claude_zustand"]    = r.get("zustand", "")
                row["claude_recycelbar"] = r.get("recycelbar", "")
                row["claude_empfehlung"] = r.get("empfehlung", "")
                row["claude_zeit"]       = round(time.time() - start, 2)
                print(f"✅ ({row['claude_zeit']}s)")
            except Exception as e:
                print(f"❌ {e}")
                row.update({k: "ERROR" for k in fieldnames if k.startswith("claude_")})

            # Llama
            print("  → Llama...", end=" ")
            try:
                start = time.time()
                r = analyze_llama(image_bytes)
                row["llama_objekt"]     = r.get("objekt", "")
                row["llama_material"]   = r.get("material", "")
                row["llama_zustand"]    = r.get("zustand", "")
                row["llama_recycelbar"] = r.get("recycelbar", "")
                row["llama_empfehlung"] = r.get("empfehlung", "")
                row["llama_zeit"]       = round(time.time() - start, 2)
                print(f"✅ ({row['llama_zeit']}s)")
            except Exception as e:
                print(f"❌ {e}")
                row.update({k: "ERROR" for k in fieldnames if k.startswith("llama_")})

            # CLIP
            print("  → CLIP...", end=" ")
            try:
                start = time.time()
                r = analyze_clip(image_bytes)
                row["clip_objekt"]     = r.get("objekt", "")
                row["clip_material"]   = r.get("material", "")
                row["clip_zustand"]    = r.get("zustand", "")
                row["clip_recycelbar"] = r.get("recycelbar", "")
                row["clip_empfehlung"] = r.get("empfehlung", "")
                row["clip_zeit"]       = round(time.time() - start, 2)
                print(f"✅ ({row['clip_zeit']}s)")
            except Exception as e:
                print(f"❌ {e}")
                row.update({k: "ERROR" for k in fieldnames if k.startswith("clip_")})

            writer.writerow(row)
            csv_file.flush()
            success += 1
            print(f"  💾 Gespeichert!")

    csv_file.close()
    print(f"\n{'='*50}")
    print(f"✅ Fertig! {success}/{total} Bilder verarbeitet.")
    print(f"📄 Ergebnisse gespeichert in: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()