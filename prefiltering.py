import pandas as pd
import re
import numpy as np
from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import langid

LEGAL_TERMS = [
    "artikel",            
    "absatz",             
    "paragraph",          
    "anlage",             
    "anhang",             
    "nummer", "nr.",      
    "kapitel",            
    "abl.",               
    "amtsblatt",          
    "eur-lex",            
]

smooth = SmoothingFunction().method1

input_file = "de-sl.tsv"
output_file = "de-sl-filtered.tsv"

min_words = 8
max_words = 100
min_len_ratio = 0.5
max_len_ratio = 2.0

remove_regulation = False
remove_template_duplicates = True
strict_legal_filter = False

langid.set_languages(['cs', 'de'])



non_latin_pattern = re.compile(r"[؀-ۿЀ-ӿ一-鿿가-힯]+")


def contains_foreign_script(text):
    if not isinstance(text, str):
        return False
    return bool(non_latin_pattern.search(text))


def is_lang(text, target_lang):
    if not isinstance(text, str):
        return False
    return langid.classify(text)[0] == target_lang


def contains_url(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"(https?://|www\.|\.[a-z]{2,4}(/|$))", text, re.IGNORECASE))


def looks_language_like(text):
    if not isinstance(text, str):
        return False
    if not text.strip():
        return False
    if re.fullmatch(r"[\W\d]+", text):
        return False
    return True


def is_regulation_like(text):
    if not isinstance(text, str):
        return False
    patterns = [
        r"č\.\s*\d+\/\d+",
        r"Nr\.?\s*\d+\/\d+",
        r"\(ES\)", r"\(EG\)",
        r"\(Úř\. věst\.\)", r"\(ABl\.\)",
        r"R\s?\d{3,5}"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def normalize_template(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d{3,}', '<NUM>', text)
    text = re.sub(r'„[^“]+“', '<QUOTE>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def has_dot_line(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"(\.\s?){3,}|(·\s?){3,}|(–\s?){3,}", text))


def contains_legal_keywords(text):
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(term in text for term in LEGAL_TERMS)


def filter_corpus(df):
    print(f"🔍 原始句对数: {len(df)}")

    df["cs"] = df["cs"].astype(str).str.strip()
    df["de"] = df["de"].astype(str).str.strip()


    df = df[df["cs"].apply(looks_language_like) & df["de"].apply(looks_language_like)]


    url_mask = df["cs"].apply(contains_url) | df["de"].apply(contains_url)
    df = df[~url_mask]
    df["cs_len"] = df["cs"].str.split().str.len()
    df["de_len"] = df["de"].str.split().str.len()
    df = df[
        (df["cs_len"] >= min_words) & (df["cs_len"] <= max_words) &
        (df["de_len"] >= min_words) & (df["de_len"] <= max_words)
    ]


    df["len_ratio"] = df["cs_len"] / df["de_len"]
    df = df[(df["len_ratio"] >= min_len_ratio) & (df["len_ratio"] <= max_len_ratio)]


   
    df = df.drop_duplicates()


    mask = df["cs"].apply(has_dot_line) | df["de"].apply(has_dot_line)
    df = df[~mask]

    before = len(df)
    df = df[
        df["cs"].apply(lambda x: is_lang(x, "cs")) &
        df["de"].apply(lambda x: is_lang(x, "de"))
    ]


    if remove_template_duplicates:
        df["cs_template"] = df["cs"].apply(normalize_template)
        df["de_template"] = df["de"].apply(normalize_template)
        template_counts = df.groupby(["cs_template", "de_template"]).size()
        duplicated = template_counts[template_counts > 1].index
        before = len(df)
        df = df[~df.set_index(["cs_template", "de_template"]).index.isin(duplicated)]


    helper_cols = ["cs_template", "de_template", "cs_len", "de_len", "len_ratio"]
    df = df.drop(columns=[c for c in helper_cols if c in df.columns])


    mask_foreign = df["cs"].apply(contains_foreign_script) | df["de"].apply(contains_foreign_script)


    df_foreign = df[mask_foreign]
    df_foreign.to_csv("de-cs-contains_foreign_script.tsv", sep="	", index=False)

    df = df[~mask_foreign]
    before = len(df)
    df = df[~df["cs"].str.contains("")]

    # 保存
    df.to_csv(output_file, sep="	", index=False)

    print(f"Final sentence pairs: {len(df)}")


def main():
    df = pd.read_csv(input_file, sep="	")
    filter_corpus(df)


if __name__ == "__main__":
    main()
