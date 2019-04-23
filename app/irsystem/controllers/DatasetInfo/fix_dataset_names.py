import json

with open("movie_dataset.json", 'r') as f:
    movie_dataset = json.load(f)

with open("../TVTropesScraper/Film/Film_tropes_dataset3.json", 'r') as f:
    movie_tropes_dataset = json.load(f)

lower_to_proper = {title.lower(): title for title in movie_tropes_dataset}

movie_dataset2 = dict()
for title in movie_dataset:
    if title.lower() in lower_to_proper:
        movie_dataset2[lower_to_proper[title.lower()]] = movie_dataset[title]
    else:
        print("Could not match {}".format(title))

with open("movie_dataset.json", 'w') as f:
    json.dump(movie_dataset2, f)
