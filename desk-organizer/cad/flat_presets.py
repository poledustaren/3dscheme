"""
Плоские профили Bambu Studio: системный профиль со всей цепочкой inherits, слитой в один JSON.

CLI Bambu Studio по одним именам профилей из проекта подставляет значения по умолчанию
(стенки 60 мм/с, плотность прутка 0), поэтому проект пересохраняется через CLI
с плоскими профилями: тогда в нём лежит полная конфигурация, как у проекта из GUI.

    python3 cad/flat_presets.py ~/opt/BambuStudio-2.08.02/resources/profiles/BBL print/presets
"""
import json
import os
import sys

PRESETS = {
    "machine": "Bambu Lab P1S 0.4 nozzle",
    "process": "0.20mm Standard @BBL X1C",
    "filament": "Bambu PLA Basic @BBL P1S 0.4 nozzle",
}


def flatten(root: str, kind: str, name: str) -> dict:
    data = json.load(open(os.path.join(root, kind, name + ".json"), encoding="utf-8"))
    parent = data.pop("inherits", "")
    base = flatten(root, kind, parent) if parent else {}
    base.update(data)
    return base


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for kind, name in PRESETS.items():
        d = flatten(root, kind, name)
        d.update({"name": name, "from": "system", "instantiation": "true"})
        path = os.path.join(out, f"{kind}.json")
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        print(path, len(d), "ключей")


if __name__ == "__main__":
    main()
