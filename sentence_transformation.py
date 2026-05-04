import random
import nltk
import spacy
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import wordnet
from nltk.tree import Tree
import pandas as pd
import re
import json
import spacy.cli

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')


spacy.cli.download("de_core_news_lg")

entity_pool = {
    "PER": [
        "Hans Müller", "Anna Schmidt", "Tobias Klein", "Laura Becker",
        "Sebastian Weber", "Julia Hoffmann", "Erik Lange"
    ],
    "LOC": [
        "Berlin", "München", "Hamburg", "Leipzig", "Köln",
        "Stuttgart", "Frankfurt", "Dresden", "Nürnberg"
    ],
    "ORG": [
        "BMW", "Volkswagen", "Siemens", "Allianz", "Deutsche Bahn",
        "SAP", "Bosch", "Adidas"
    ]
}

with open("antonyms_de.json", "r", encoding="utf-8") as f:
    antonyms_dict = json.load(f)

SPELLED_NUMBERS = [
    "eins", "zwei", "drei", "vier", "fünf", "sechs",
    "sieben", "acht", "neun", "zehn", "elf", "zwölf","ein"
]
_YEAR_RE = re.compile(r'^(1[0-9]{3}|20[0-1][0-9])$')
_NUM_RE  = re.compile(r'^\d+(?:[.,]\d+)*$')
_SPLIT_RE = re.compile(r'^(\D*)(\d+(?:[.,]\d+)*)(\D*)$')

nlp = spacy.load("de_core_news_lg")


def toggle_negation(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    if "nicht" in tokens:
        tokens = [t for t in tokens if t.lower() != "nicht"]
        return " ".join(tokens)
    if any(t.lower().startswith("kein") for t in tokens):
        tokens = [t for t in tokens if not t.lower().startswith("kein")]
        return " ".join(tokens)
    
    for i, token in enumerate(doc):
        if token.pos_ in {"VERB", "AUX"}:
            tokens.insert(i + 1, "nicht")
            return " ".join(tokens)

    return " ".join(tokens)


def strengthen_modality_de(text):
    mapping = {
    # können → müssen
    "kann": "muss",
    "können": "müssen",
    "könnt": "müsst",
    "konnte": "musste",
    "konnten": "mussten",
    "konntest": "musstest",
    "konntet": "musstet",
    "könnte": "müsste",
    "könnten": "müssten",

    # dürfen → werden
    "darf": "wird",
    "dürfen": "werden",
    "dürft": "werdet",
    "durfte": "wurde",
    "durften": "wurden",
    "durftest": "wurdest",
    "durftet": "wurdet",
    "dürfte": "würde",
    "dürften": "würden",

    # sollen → werden
    "soll": "wird",
    "sollen": "werden",
    "sollt": "werdet",
    "sollte": "würde",
    "sollten": "würden",
    "solltest": "würdest",
    "solltet": "würdet",

    # mögen → werden
    "mag": "wird",
    "mögen": "werden",
    "mögt": "werdet",
    "mochte": "wurde",
    "mochten": "wurden",
    "mochtest": "wurdest",
    "mochtet": "wurdet",
    "möchte": "würde",
    "möchten": "würden"
    }
    doc = nlp(text)
    new_words = []
    for token in doc:
        lemma = token.lemma_
        if lemma in mapping:
            replacement = mapping[lemma]
            if token.text[0].isupper():
                replacement = replacement.capitalize()
            new_words.append(replacement)
        else:
            new_words.append(token.text)
    return " ".join(new_words)

def replace_adjective_antonyms(text):
    words = text.split()
    new_words = []
    replaced = False

    for word in words:
        lower = word.lower()
        for adj, antonym in antonyms_dict.items():
            if lower.startswith(adj):
                suffix = lower[len(adj):]
                new_word = antonym + suffix
                if word[0].isupper():
                    new_word = new_word.capitalize()
                new_words.append(new_word)
                replaced = True
                break
        else:
            new_words.append(word)

    return " ".join(new_words) if replaced else text  


def entity_replacement(text):
    doc = nlp(text)
    for ent in doc.ents:
        entity_type = ent.label_
        if entity_type in entity_pool and entity_pool[entity_type]:
            replacement = random.choice(entity_pool[entity_type])
            return text[:ent.start_char] + replacement + text[ent.end_char:]
    return text

def contains_modality(text):
    doc = nlp(text)
    modals = {"können", "dürfen", "sollen", "mögen", "müssen", "wollen"}
    return any(token.lemma_ in modals and token.pos_ == "AUX" for token in doc)

def can_negation(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]
    has_verb = any(token.pos_ in {"VERB", "AUX"} for token in doc)
    return has_verb

def contains_adjective_with_antonym(text):
    words = text.lower().split()
    return any(w in antonyms_dict for w in words)

def filter_for_number(sentence):
    return bool(re.search(r'\b\d+%?|\b(eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)\b', sentence.lower()))

def filter_for_entity(sentence):
    doc = nlp(sentence)
    return any(ent.label_ in {"PER", "LOC", "ORG"} for ent in doc.ents)

def filter_for_causality(text):
    return (
        contains_modality(text)
        or can_negation(text)
        or contains_adjective_with_antonym(text)
    )

def number_replacement(text):
    doc = nlp(text)
    new_tokens = []
    
    for token in doc:
        txt   = token.text
        lower = txt.lower()
        m     = _SPLIT_RE.match(txt)

        if lower in SPELLED_NUMBERS:
            new_tokens.append(random.choice(SPELLED_NUMBERS))
            continue

        if m:
            prefix, num_str, suffix = m.groups()

            if _YEAR_RE.match(num_str):
                orig_year = int(num_str)
                new_num   = str(random.randint(1900, 2025))

            elif _NUM_RE.match(num_str):
                norm = num_str.replace(',', '.')
                try:
                    val = float(norm)
                except ValueError:
                    val = None

                if val is not None and val > 1000:
                    new_num = str(random.randint(1001, 100000))
                elif val is not None and val % 1 != 0:
                    new_num = str(round(random.uniform(1, 100), 2)).replace('.', ',')
                else:
                    new_num = str(random.randint(1, 100))
            else:
                new_tokens.append(txt)
                continue

            new_tokens.append(f"{prefix}{new_num}{suffix}")

        else:
            new_tokens.append(txt)
            
    return " ".join(new_tokens)

def replace_surface_tokens(input_file, sample_size=15000):
    df = pd.read_csv(input_file, sep="\t")

    transformations = {
        "entity": {
            "func": entity_replacement,
            "file": "transformed_entity_cs.tsv",
            "filter": filter_for_entity
        },
        "number": {
            "func": number_replacement,
            "file": "transformed_number_cs.tsv",
            "filter": filter_for_number
        }
    }
    print("Start")


    all_augmented = []
    print("Loading")

    for i,(name, info) in enumerate(transformations.items()):
        print(f"▶ 正在处理: {name} ...")
        func       = info["func"]
        output_file= info["file"]
        filter_func= info["filter"]
        augmented_rows = []
        df_s = df.sample(frac=1, random_state=42 + i).reset_index(drop=True)

        for _, row in df_s.iterrows():
            if len(augmented_rows) >= sample_size:
                break

            src = row["sl"]
            tgt = row["de"]
         
            if not filter_func(tgt):
                continue
            new_tgt = func(tgt)
     
            if new_tgt != tgt:
                augmented_rows.append({
                    "sl": src,
                    "de": new_tgt,
                    "label": 0,
                    "transformation": name
                    })



        actual = len(augmented_rows)
        if actual < sample_size:
            print(f"{name} has not enough samples {actual}/{sample_size}")

        out_df = pd.DataFrame(augmented_rows)
        out_df.to_csv(output_file, sep="\t", index=False, header=True)

def logical_relation(input_file, sample_size=10000):
    df = pd.read_csv(input_file, sep="\t")
    causality_modes = {
    "modality": {
        "filter": contains_modality,
        "func": strengthen_modality_de,
        "file": "cs-transformed_causality_modality.tsv"
    },
    "negation": {
        "filter": can_negation,
        "func": toggle_negation,
        "file": "cs-transformed_causality_negation.tsv"

    },
    "antonym": {
        "filter": contains_adjective_with_antonym,
        "func": replace_adjective_antonyms,
        "file": "cs-transformed_causality_antonym.tsv"

    }
}

    for i, (mode, info) in enumerate(causality_modes.items()):
        print(f"causality: {mode}")
        func = info["func"]
        filter_func = info["filter"]
        output_file = info["file"]
        count = 0
        causality_augmented_rows = []
        df_shuffled = df.sample(frac=1, random_state=80+i).reset_index(drop=True)

        for _, row in df_shuffled.iterrows():
            if count >= sample_size:
                break

            src = row["cs"]
            tgt = row["de"]

            if not filter_func(tgt):
                continue

            new_tgt = func(tgt)

            if new_tgt != tgt:
                causality_augmented_rows.append({
                    "cs": src,
                    "de": new_tgt,
                    "label": 0,
                    "transformation": mode
                })
                count += 1
    
        actual = len(causality_augmented_rows)
        if actual < sample_size:
            print(f"{mode} has not enough samples {actual}/{sample_size}")

        out_df = pd.DataFrame(causality_augmented_rows)
        out_df.to_csv(output_file, sep="\t", index=False, header=True)

 



if __name__ == "__main__":
    replace_surface_tokens(
        input_file="de-cs-filtered.tsv",
        sample_size=15000,
    )
    logical_relation(
        input_file="de-cs-filtered.tsv",
        sample_size=10000,
    )