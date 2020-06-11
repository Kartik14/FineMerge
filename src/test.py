from utils import normalize_string
import json 

with open('labels_char.json') as fd:
    labels = json.load(fd)

print(labels)
print(normalize_string('call me at 8:00 pm!', labels[1:]))