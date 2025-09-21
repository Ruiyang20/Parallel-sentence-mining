import pandas as pd
import re
import numpy as np
from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import langid

LEGAL_TERMS = [
    "artikel",            # 法律条款
    "absatz",             # 段落
    "paragraph",          # 段落 §
    "anlage",             # 附件
    "anhang",             # 附录
    "nummer", "nr.",      # 编号 Nr. 1234/2010
    "kapitel",            # 章节
    "abl.",               # ABl. = Amtsblatt = 官方公报
    "amtsblatt",          # 全称
    "eur-lex",            # 欧盟法库
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

langid.set_languages(['sl', 'de'])


# 匹配非拉丁字符（如阿拉伯、波斯、汉字、韩文等）
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
    # 匹配 3 组以上的点号或中点（英文句点或 Unicode）
    return bool(re.search(r"(\.\s?){3,}|(·\s?){3,}|(–\s?){3,}", text))


def contains_legal_keywords(text):
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(term in text for term in LEGAL_TERMS)


def filter_corpus(df):
    print(f"🔍 原始句对数: {len(df)}")

    df["sl"] = df["sl"].astype(str).str.strip()
    df["de"] = df["de"].astype(str).str.strip()

    # 1.  语言内容过滤
    df = df[df["sl"].apply(looks_language_like) & df["de"].apply(looks_language_like)]

    print(f"✅ 语言内容过滤后: {len(df)}")
    # 2. 网址过滤
    url_mask = df["sl"].apply(contains_url) | df["de"].apply(contains_url)
    print(f"❌ 删除包含 URL 的句对: {url_mask.sum()}")
    df = df[~url_mask]
    # 2. 最小词数过滤
    df["sl_len"] = df["sl"].str.split().str.len()
    df["de_len"] = df["de"].str.split().str.len()
    df = df[
        (df["sl_len"] >= min_words) & (df["sl_len"] <= max_words) &
        (df["de_len"] >= min_words) & (df["de_len"] <= max_words)
    ]

    print(f"✅ 词数过滤后: {len(df)}")

    # 3. 长度比例过滤
    df["len_ratio"] = df["sl_len"] / df["de_len"]
    df = df[(df["len_ratio"] >= min_len_ratio) & (df["len_ratio"] <= max_len_ratio)]
    print(f"✅ 长度比例过滤后: {len(df)}")

    # 4. 精确重复句对过滤
    df = df.drop_duplicates()
    print(f"✅ 去重后: {len(df)}")

    # 5. 占位符样式句子过滤（点点点）
    mask = df["sl"].apply(has_dot_line) | df["de"].apply(has_dot_line)
    print(f"❌ 删除占位符样式句子数: {mask.sum()}")
    df = df[~mask]

    before = len(df)
    df = df[
        df["sl"].apply(lambda x: is_lang(x, "sl")) &
        df["de"].apply(lambda x: is_lang(x, "de"))
    ]
    print(f"❌ 其他语言 {before - len(df)}")

    # 6. 模板重复句过滤（结构雷同）
    if remove_template_duplicates:
        df["sl_template"] = df["sl"].apply(normalize_template)
        df["de_template"] = df["de"].apply(normalize_template)
        template_counts = df.groupby(["sl_template", "de_template"]).size()
        duplicated = template_counts[template_counts > 1].index
        before = len(df)
        df = df[~df.set_index(["sl_template", "de_template"]).index.isin(duplicated)]
        print(f"❌ 删除模板重复句对: {before - len(df)}")

    helper_cols = ["sl_template", "de_template", "sl_len", "de_len", "len_ratio"]
    df = df.drop(columns=[c for c in helper_cols if c in df.columns])

    # 检测 cs 和 de 是否包含外语字符（如波斯语、阿拉伯语）
    mask_foreign = df["sl"].apply(contains_foreign_script) | df["de"].apply(contains_foreign_script)

    # 输出这些包含嵌入外语的句子（可手动检查）
    df_foreign = df[mask_foreign]
    df_foreign.to_csv("de-sl-contains_foreign_script.tsv", sep="	", index=False)
    print(f"❌ 嵌入其他语言: {len(df_foreign)}")
    df = df[~mask_foreign]
    before = len(df)
    df = df[~df["sl"].str.contains("�")]
    print(f"❌ 含有无法识别符号: {before - len(df)}")
    # 保存
    df.to_csv(output_file, sep="	", index=False)

    print(f"✅ 最终保留句对数: {len(df)}")


def main():
    df = pd.read_csv(input_file, sep="	")
    filter_corpus(df)


if __name__ == "__main__":
    main()
