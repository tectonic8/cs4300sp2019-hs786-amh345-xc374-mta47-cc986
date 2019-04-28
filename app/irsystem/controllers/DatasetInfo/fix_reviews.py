import json

with open("book_dataset.json", 'r') as f:
    dataset = json.load(f)

for k in dataset:
    if "reviews" not in dataset[k]: continue
    for r in dataset[k]["reviews"]:
        r["text"] = r["text"].replace("<b>", " <b>").strip()

with open("book_dataset.json", 'w') as f:
    json.dump(dataset, f)