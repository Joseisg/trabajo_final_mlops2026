"""
Genera defaults_modelos.json con los valores más comunes de
combustible, transmisión y carrocería por combinación marca+modelo.
"""
import pandas as pd
import json

df = pd.read_csv("ml_raw_detalle.csv", usecols=[
    "marca", "modelo", "tipo_de_combustible", "transmision", "tipo_de_carroceria"
], low_memory=False)

df = df.dropna(subset=["marca", "modelo"])

defaults = {}
for (marca, modelo), group in df.groupby(["marca", "modelo"]):
    key = f"{marca}|{modelo}"
    defaults[key] = {
        "tipo_de_combustible": group["tipo_de_combustible"].mode().iloc[0] if group["tipo_de_combustible"].notna().any() else "Bencina",
        "transmision": group["transmision"].mode().iloc[0] if group["transmision"].notna().any() else "Automática",
        "tipo_de_carroceria": group["tipo_de_carroceria"].mode().iloc[0] if group["tipo_de_carroceria"].notna().any() else "SUV",
    }

with open("defaults_modelos.json", "w", encoding="utf-8") as f:
    json.dump(defaults, f, ensure_ascii=False, indent=2)

print(f"[OK] defaults_modelos.json generado con {len(defaults)} combinaciones")
