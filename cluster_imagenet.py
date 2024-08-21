from nltk.corpus import wordnet as wn
import yaml
from collections import deque
from tqdm import tqdm

import nltk
nltk.download('wordnet')

# def get_hyponyms_dict(synset):
#     """Recursively get hyponyms of a synset in a nested dictionary structure."""
#     hyponyms_dict = {}
#     for hyponym in synset.hyponyms():
#         hyponym_lemmas = ', '.join(lemma.name() for lemma in hyponym.lemmas())
#         hyponyms_dict[hyponym_lemmas] = get_hyponyms_dict(hyponym)
#     return hyponyms_dict
#
# def get_noun_hyponyms_dict(word):
#     """Get all noun hyponyms of the first noun synset of a given word in a nested dictionary structure."""
#     hyponyms_dict = {}
#     synsets = wn.synsets(word, pos=wn.NOUN)
#     if synsets:
#         first_synset = synsets[0]
#         synset_name = first_synset.name().split('.')[0]
#         hyponyms_dict[synset_name] = get_hyponyms_dict(first_synset)
#     return hyponyms_dict
#
# word = "dog"  # Replace this with the word you want to find hyponyms for
# hyponyms_dict = get_noun_hyponyms_dict(word)
#
# import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(hyponyms_dict)

###############################################################

def get_hyponyms(word):
    """Get hyponyms for a given word using WordNet."""
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
    for word in tqdm(words):

        #direct_hyponyms = get_hyponyms(word)
        hyponyms = get_all_hyponyms(word)
        #hyponyms = get_klevel_hyponyms(word, 2)

        #filtering out words that are not in the list
        # filter = []
        # for hyponym in hyponyms:
        #     if hyponym not in words:
        #         filter.append(False)
        #     else:
        #         filter.append(True)
        # hyponyms = [hyponyms[i] for i in range(len(hyponyms)) if filter[i]]

        hierarchy[word] = [hyponym.lower() for hyponym in hyponyms]
    return hierarchy

#recursive

def get_hyponyms_rec(word):
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

# Assuming the rest of your code for loading YAML files and extracting class names remains unchanged

# Example loading YAML files and extracting class names (replace with your actual file paths and loading logic)
with open('./data_class_lists/imagenet_cls.yaml', 'r') as file:
    list2 = yaml.safe_load(file)
class_names_list2 = list(list2.values())
class_names_list2 = [name.lower() for name in class_names_list2]

# Build the hierarchy with direct hyponyms
#hierarchy = build_hierarchy(class_names_list2)

# Build the hierarchy with iterative hyponyms
hierarchy = build_hierarchy(class_names_list2)


import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(hierarchy)

#save hierarchy to file
with open('data_class_lists/imagenet_cls_hierarchy.yaml', 'w') as file:
    yaml.dump(hierarchy, file)
