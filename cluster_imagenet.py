from nltk.corpus import wordnet as wn
import yaml
from collections import deque
from tqdm import tqdm
import os, sys

import nltk
nltk.download('wordnet')

def get_hyponyms(word):
    """Get first level hyponyms for a given word using WordNet."""
    synsets = wn.synsets(word)
    if not synsets:
        return []
    hyponyms = synsets[0].hyponyms()  # Fetching hyponyms instead of hypernyms
    return [hyponym.lemma_names()[0] for hyponym in hyponyms]

def get_all_hyponyms(word):
    """Get all hyponyms for a given word using WordNet."""
    synsets = wn.synsets(word)
    if not synsets:
        return []
    all_hyponyms = set()
    for synset in synsets:
        hyponyms = synset.hyponyms()
        for hyponym in hyponyms:
            all_hyponyms.update(hyponym.lemma_names())
    return list(all_hyponyms)

def get_klevel_hyponyms(word, k):
    """Get k-level hyponyms for a given word using WordNet."""
    synsets = wn.synsets(word)
    if not synsets:
        return []
    hyponyms = set()
    for synset in synsets:
        hyponyms.update(synset.hyponyms())
    for _ in range(k - 1):
        next_level_hyponyms = set()
        for hyponym in hyponyms:
            next_level_hyponyms.update(hyponym.hyponyms())
        hyponyms = next_level_hyponyms
    return [hyponym.lemma_names()[0] for hyponym in hyponyms]

def build_hierarchy(words):
    """Build a hierarchical structure using hyponyms."""
    hierarchy = {}
    for word in tqdm(words, desc='Building hierarchy'):
        hyponyms = get_all_hyponyms(word)
        hierarchy[word] = [hyponym.lower() for hyponym in hyponyms]
    return hierarchy

def build_hierarchy_rec(words):
    """Build a hierarchical structure using hyponyms iteratively."""
    hierarchy = {}
    queue = deque([(word, hierarchy) for word in words])
    cont = 0

    while queue:
        cont += 1
        current_word, current_hierarchy = queue.popleft()
        hyponyms = get_hyponyms(current_word)
        if hyponyms:
            current_hierarchy[current_word] = {hyponym.lower(): {} for hyponym in hyponyms}
            for hyponym in hyponyms:
                queue.append((hyponym.lower(), current_hierarchy[current_word][hyponym.lower()]))
        if cont % 100 == 0:
            print(f'Queue size {len(queue)}')

    return hierarchy

if __name__ == '__main__':

    print("Building hierarchy for ImageNet classes...")

    out_file = './data_class_lists/imagenet_cls_hierarchy.yaml'

    # Loading YAML files and extracting class names
    input_file = './data_class_lists/imagenet_cls.yaml'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found, please check that the file with imagenet synsets and class names is in the correct path.")

    with open(input_file, 'r') as file:
        list2 = yaml.safe_load(file)
    class_names_list2 = list(list2.values())
    class_names_list2 = [name.lower() for name in class_names_list2]

    # Build the hierarchy with iterative hyponyms
    hierarchy = build_hierarchy(class_names_list2)

    #save hierarchy to file
    with open(out_file, 'w') as file:
        yaml.dump(hierarchy, file)
        print(f"Saved hierarchy to {out_file}")
