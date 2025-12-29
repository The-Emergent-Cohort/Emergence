# MMoOn - Multilingual Morpheme Ontology

## Overview

MMoOn is the first comprehensive ontology for morphological language data. It captures semantics at the sub-word level with machine-processable morpheme inventories.

**Key Structure:**
- **Morpheme** = atomic concept unit (abstract)
- **Morph** = surface form (concrete, language-specific)
- **Meaning** = structured semantic definition with English glosses

## Available Datasets

| Language | Entries | File | Status |
|----------|---------|------|--------|
| Hebrew | ~52,000 | `heb/inventory/oh.ttl` | Most complete |
| German | Sample | `deu/inventory/*.ttl` | Partial |
| Xhosa (Bantu) | WIP | `xho/inventory/*.ttl` | Work in progress |

**All datasets include English translations/glosses.**

Xhosa specifically notes: "lexical and morphological data, English translations and is linked to WordNet RDF"

## Download

```bash
# Clone all three
git clone https://github.com/MMoOn-Project/OpenHebrew.git
git clone https://github.com/MMoOn-Project/OpenGerman.git
git clone https://github.com/MMoOn-Project/OpenBantu.git

# Or use curation repo with submodules
git clone --recursive https://github.com/MMoOn-Project/curation.git
```

## Data Format

Turtle (.ttl) - RDF triples, text-based

**Key properties for English glosses:**
- `rdfs:label`
- `lemon:sense`
- `skos:definition`
- `mmoon:meaning`

## Ontology Structure

```
Morpheme (abstract concept)
├── Root
├── Stem
└── Affix (Prefix, Suffix, Infix)

Morph (surface forms) → correspondsToMorpheme → Morpheme → hasMeaning → Meaning
```

**430 classes, 37 object properties, 5 datatype properties**

Integrates with:
- OLiA (Ontologies of Linguistic Annotation) for POS/grammar
- OntoLex-Lemon for lexical data
- WordNet RDF

## Use Case for Tokenizer DB

Extract English glosses from Hebrew/German/Xhosa morphemes to build concept vocabulary:

```
Hebrew morpheme → English gloss → concept
German morpheme → English gloss → concept (validates)
Xhosa morpheme → English gloss → concept (validates)
```

Cross-language validation: if Hebrew "-im" and German "-en" both gloss to "plural", confirms PLURAL as core concept.

## Resources

- **Website**: https://mmoon.org/ (may be down)
- **SPARQL**: https://mmoon.org/sparql/
- **GitHub**: https://github.com/MMoOn-Project
- **Core Paper**: Klimek et al., 2021 - "MMoOn Core – the Multilingual Morpheme Ontology"
- **Hebrew Paper**: https://aclanthology.org/L16-1143/

## Project Status

Most repos inactive since 2018-2020. Data is stable but not actively maintained.

## Comparison to Wiktionary Etymology

| Aspect | Wiktionary | MMoOn |
|--------|-----------|-------|
| Granularity | Word-form level | Segmented morpheme level |
| Morpheme Decomposition | Minimal | Explicit morpheme-morph relationships |
| Meaning Representation | Free-text definitions | Structured semantic annotations |
| Formal Semantics | Free-text | Machine-processable RDF |
| English Glosses | Variable | Standardized |

**MMoOn provides granular morpheme-level decomposition; Wiktionary provides word-level annotations.**
