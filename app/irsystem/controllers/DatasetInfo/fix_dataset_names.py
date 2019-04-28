import json

dataset_type = "movie"

if dataset_type == "movie":

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

    idx = 0
    for k in movie_dataset2:
        movie_dataset2[k]["idx"] = idx
        idx += 1
    print(idx)

    with open("movie_dataset.json", 'w') as f:
        json.dump(movie_dataset2, f)
else:
    with open("book_dataset.json", 'r') as f:
        book_dataset = json.load(f)

    with open("../TVTropesScraper/Literature/Literature_tropes_dataset3.json", 'r') as f:
        book_tropes_dataset = json.load(f)

    lower_to_proper = {title.lower(): title for title in book_tropes_dataset}

    book_dataset2 = dict()
    for title in book_dataset:
        if title.lower() in lower_to_proper:
            book_dataset2[lower_to_proper[title.lower()]] = book_dataset[title]
        else:
            print("Could not match {}".format(title))

    idx = 0
    for k in book_dataset2:
        book_dataset2[k]["idx"] = idx
        idx += 1
    print(idx)
    with open("book_dataset.json", 'w') as f:
        json.dump(book_dataset2, f)