# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Learning from the worst Hate Speech dataset."""


import csv

import datasets


_DESCRIPTION = """\
We present a human-and-model-in-the-loop process for dynamically generating datasets and training better performing and more robust hate detection models. We provide a new dataset of ~40,000 entries, generated and labelled by trained annotators over four rounds of dynamic data creation. It includes ~15,000 challenging perturbations and each hateful entry has fine-grained labels for the type and target of hate. Hateful entries make up 54% of the dataset, which is substantially higher than comparable datasets. We show that model performance is substantially improved using this approach. Models trained on later rounds of data collection perform better on test sets and are harder for annotators to trick. They also perform better on HATECHECK, a suite of functional tests for online hate detection. See https://arxiv.org/abs/2012.15761 for more details.
"""

_CITATION = """\
@inproceedings{vidgen2021learning,
  title={Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection},
  author={Vidgen, Bertie and Thrush, Tristan and Waseem, Zeerak and Kiela, Douwe},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={1667--1682},
  year={2021}
}
"""

_DOWNLOAD_URL = "https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/main/Dynamically%20Generated%20Hate%20Dataset%20v{version}.csv"
_VERSIONS = ("0.2.3", "0.2.2")


class Dynahate(datasets.GeneratorBasedBuilder):
    """Learning from the worst hate speech classification dataset."""
    BUILDER_CONFIGS = [datasets.BuilderConfig(name=version, version=version) for version in _VERSIONS]
    DEFAULT_CONFIG_NAME = "0.2.3"
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "acl.id": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["nothate", "hate"]),
                    "text": datasets.Value("string"),
                    "X1": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "level": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "round.base": datasets.Value("int32"),
                    "annotator": datasets.Value("string"),
                    "round": datasets.Value("string"),
                    "acl.id.matched": datasets.Value("string")
                }
            ),
            homepage="https://arxiv.org/abs/2012.15761",
            citation=_CITATION,
        )


    def _generate_examples(self, csv_path, split):
        """Generate AG News examples."""
        with open(csv_path, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    keys = row[:]
                else:
                    res = dict([(k, v) for k, v in zip(keys, row) if k != ""])
                
                    for k in ["X1", "round.base"]:
                        res[k] = int(res[k])
                    yield id_ - 1, res