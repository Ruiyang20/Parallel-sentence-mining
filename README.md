# Parallel-sentence-mining


This repository contains the core scripts used in the thesis:  
**Multilingual Embedding-Based Sentence Pair Analysis in Low-Resource Languages**.

---

## Files
- `sentence_embedding.py` — encode sentences with multilingual models (XLM-R, Glot500m, LaBSE)  
- `sentence_transformation.py` — apply transformations (negation, antonym, etc.)  
- `cbie_main.py` — run experiments with CBIE  
- `cbie_transformation.py` — CBIE-related preprocessing  
- `cbie_visual.py` — visualization of embedding spaces  
- `evaluation.py` — compute classification metrics (accuracy, precision, recall, F1, etc.)  
- `plot_curve.py` — plotting curves (e.g., ROC, PR)  
- `antonyms_de.json` — resource file for antonym-based transformations  
