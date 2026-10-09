"""Shared academic entity schema and prompts for graph extraction.

Neo4j doc-level extraction and LightRAG chunk-level extraction both import this
module so entity types and anti-hallucination rules stay in one place.
"""

from __future__ import annotations

from typing import Literal, TypedDict

AcademicEntityType = Literal["Method", "Dataset", "Metric", "Task"]

ACADEMIC_ENTITY_TYPES: tuple[AcademicEntityType, ...] = (
    "Method",
    "Dataset",
    "Metric",
    "Task",
)

ACADEMIC_ENTITY_TYPE_DESCRIPTIONS: dict[AcademicEntityType, str] = {
    "Method": (
        "Specific named method, model, or algorithm such as STGCN, BERT, "
        "Diffusion Model. DO NOT extract generic terms like neural network."
    ),
    "Dataset": (
        "Specific named dataset such as PEMS-BAY, ImageNet, GLUE. "
        "DO NOT extract generic terms like the dataset or image dataset."
    ),
    "Metric": (
        "Specific named metric such as MAPE, F1, BLEU, Accuracy. "
        "DO NOT extract vague terms like performance or accuracy improvement."
    ),
    "Task": (
        "Specific named research task such as traffic forecasting or image "
        "classification. DO NOT extract broad topics like AI or deep learning."
    ),
}


class AcademicEntity(TypedDict):
    canonical_name: str
    type: AcademicEntityType
    aliases: list[str]
    evidence_excerpt: str


ACADEMIC_DOC_EXTRACTION_PROMPT = """
You are an academic paper knowledge extraction system. Analyze the paper
content and extract significant entities of 4 specific types only.

<paper>
<content>
{{full_text}}
</content>
</paper>

<instructions>
Extract entities of these 4 types ONLY:
- Method: {method_description}
- Dataset: {dataset_description}
- Metric: {metric_description}
- Task: {task_description}

Frequency rule:
Only extract entities that appear at least twice in the content OR are
described in detail in a dedicated paragraph, table, or bullet list. A short
abstract may count when the entity is central to the paper.

Empty content rule:
If content is empty, contains only image references, or has no substantive
textual information, output exactly: {{"entities": []}}. Do not invent
entities from filenames, titles, citations, affiliations, or external clues.

Anti-noise rules:
- DO NOT extract author names, cited author names, location names, paper
  titles, organization names, affiliation names, or generic concepts.
- DO NOT extract parent categories, sub-modules, related techniques, or
  implementation details as aliases.
- Each item must belong to exactly one type. Do not duplicate the same entity
  across types.

Aliases rule:
For each entity, aliases may include only official abbreviations, full or short
name variants, spelling variants, or translations used interchangeably for the
same thing.

JSON format:
Output ONLY valid JSON. No preamble, markdown, comments, or trailing text.
Every entity must contain canonical_name, type, aliases, and evidence_excerpt.

Output schema:
{{
  "entities": [
    {{
      "canonical_name": "STGCN",
      "type": "Method",
      "aliases": ["Spatio-Temporal Graph Convolutional Network"],
      "evidence_excerpt": "We propose STGCN, a spatio-temporal graph convolutional network..."
    }}
  ]
}}
</instructions>
""".strip().format(
    method_description=ACADEMIC_ENTITY_TYPE_DESCRIPTIONS["Method"],
    dataset_description=ACADEMIC_ENTITY_TYPE_DESCRIPTIONS["Dataset"],
    metric_description=ACADEMIC_ENTITY_TYPE_DESCRIPTIONS["Metric"],
    task_description=ACADEMIC_ENTITY_TYPE_DESCRIPTIONS["Task"],
)

ACADEMIC_ENTITY_EXTRACTION_SYSTEM_PROMPT = """
---Role---
You are an academic paper Knowledge Graph Specialist. You extract ONLY these
entity types from research paper text: {entity_types}.

---Instructions---
1. Entity types are CLOSED: only output Method, Dataset, Metric, or Task.
   Anything else is Other and MUST NOT be output.
2. Specific names only: extract named methods, datasets, metrics, and research
   tasks. DO NOT extract generic terms such as neural network, the dataset,
   performance, deep learning, author names, location names, affiliations, or
   paper titles.
3. Frequency rule: skip one-off mentions. An entity must appear at least twice
   OR be discussed in detail.
4. Empty content rule: if the input is empty or image-only, return no entity
   records.
5. Extract relationships only between valid academic entities when the text
   clearly states usage, evaluation, comparison, or task focus.
6. Use the requested output language for descriptions. Keep entity names in
   their original scholarly spelling.
7. Preserve LightRAG tuple format exactly:
   entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description
   relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}keywords{tuple_delimiter}description
8. Finish with {completion_delimiter}.

---Examples---
{examples}
""".strip()

ACADEMIC_ENTITY_EXTRACTION_EXAMPLES: list[str] = [
    """
<Entity_types>
["Method","Dataset","Metric","Task"]

<Input Text>
```
STGCN is proposed for traffic forecasting on the PeMSD7 benchmark. The STGCN
model uses spatio-temporal graph convolution and is evaluated with MAE, RMSE,
and MAPE on PeMSD7.
```

<Output>
entity{tuple_delimiter}STGCN{tuple_delimiter}Method{tuple_delimiter}STGCN is a spatio-temporal graph convolutional network proposed for traffic forecasting.
entity{tuple_delimiter}PeMSD7{tuple_delimiter}Dataset{tuple_delimiter}PeMSD7 is the benchmark dataset used to evaluate STGCN.
entity{tuple_delimiter}MAPE{tuple_delimiter}Metric{tuple_delimiter}MAPE is one of the evaluation metrics reported for STGCN.
entity{tuple_delimiter}traffic forecasting{tuple_delimiter}Task{tuple_delimiter}Traffic forecasting is the research task studied by the paper.
relation{tuple_delimiter}STGCN{tuple_delimiter}PeMSD7{tuple_delimiter}evaluation dataset{tuple_delimiter}STGCN is evaluated on PeMSD7.
relation{tuple_delimiter}STGCN{tuple_delimiter}traffic forecasting{tuple_delimiter}studies task{tuple_delimiter}STGCN is proposed for traffic forecasting.
{completion_delimiter}
""".strip(),
    """
<Entity_types>
["Method","Dataset","Metric","Task"]

<Input Text>
```
BERT is fine-tuned for natural language inference and question answering.
Experiments on GLUE and SQuAD report Accuracy and F1. The transformer encoder
is used as part of BERT, but generic phrases such as deep learning are ignored.
```

<Output>
entity{tuple_delimiter}BERT{tuple_delimiter}Method{tuple_delimiter}BERT is the named model fine-tuned for language understanding tasks.
entity{tuple_delimiter}GLUE{tuple_delimiter}Dataset{tuple_delimiter}GLUE is a benchmark dataset used in the experiments.
entity{tuple_delimiter}SQuAD{tuple_delimiter}Dataset{tuple_delimiter}SQuAD is a benchmark dataset used for question answering experiments.
entity{tuple_delimiter}F1{tuple_delimiter}Metric{tuple_delimiter}F1 is reported as an evaluation metric.
entity{tuple_delimiter}question answering{tuple_delimiter}Task{tuple_delimiter}Question answering is one of the research tasks evaluated.
relation{tuple_delimiter}BERT{tuple_delimiter}SQuAD{tuple_delimiter}evaluation dataset{tuple_delimiter}BERT is evaluated on SQuAD.
relation{tuple_delimiter}BERT{tuple_delimiter}question answering{tuple_delimiter}studies task{tuple_delimiter}BERT is fine-tuned for question answering.
{completion_delimiter}
""".strip(),
]
