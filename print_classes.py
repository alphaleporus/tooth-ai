import json
data = json.load(open("data/final-di/train/_annotations.coco.json", "r"))
cats = sorted(data["categories"], key=lambda x: x["id"])
for c in cats:
    print(f"{c['id']:4d}  {c['name']}")
