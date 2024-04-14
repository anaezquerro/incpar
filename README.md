# :pencil: [IncPar](https://github.com/anaezquerro/incpar): Fully [Inc](https://github.com/anaezquerro/incpar)remental Neural Dependency and Constituency [Par](https://github.com/anaezquerro/incpar)sing

A Python package for reproducing results of fully incremental dependency and constituency parsers described in:

- [On The Challenges of Fully Incremental Neural Dependency Parsing](https://aclanthology.org/2023.ijcnlp-short.7/) at [IJCNLP-AACL 2023](http://www.ijcnlp-aacl2023.org/).
- [From Partial to Strictly Incremental Constituent Parsing](https://aclanthology.org/2024.eacl-short.21/) at [EACL 2024](https://2024.eacl.org/).
- [Fully Incremental Parsing based on Neural Networks](https://ruc.udc.es/dspace/handle/2183/33269).

**Note**: Our implementation was built from forking [yzhangcs](https://github.com/yzhangcs)' [SuPar v1.1.4](https://github.com/yzhangcs/parser) repository. The Vector Quantization module was extracted from [lucidrains](https://github.com/lucidrains)' [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and Sequence Labeling encodings from [Polifack](https://github.com/Polifack)'s [CoDeLin](https://github.com/Polifack/codelin) repositories.

## Incremental Parsers

* Dependency Parsing:
    * [Sequence Labeling](https://aclanthology.org/N19-1077/) (absolute, relative, PoS-based and bracketing encodings).
    * [Transition-based w. Arc-Eager](https://aclanthology.org/C12-1059/).
* Constituency Parsing:
    * [Sequence Labeling](https://aclanthology.org/D18-1162/) (absolute and relative encodings).
    * [Attach-Juxtapose](https://arxiv.org/abs/2010.14568).


## Usage

In order to reproduce our experiments, follow the installation and deployment steps of [SuPar](https://github.com/yzhangcs/parser), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and [CoDeLin](https://github.com/Polifack/codelin) repositories. Supported functionalities are **training**, **evaluation** and **prediction** from CoNLL-U or PTB-bracketed files. We highly suggest to run our parsers using terminal commands in order to train and generate prediction files. In the future :raised_hands: we'll make available [SuPar methods](https://github.com/yzhangcs/parser#usage) to easily test our parsers' performance from Python terminal.

### Training

**Dependency Parsing**:

* **Sequence labeling Dependency Parser** ([`SLDependencyParser`](supar/models/dep/sl/parser.py)): Inherits all arguments of the main class [`Parser`](supar/parser.py) and allows the flag `--codes` to specify encoding to configure the trees linearization (`abs`, `rel`, `pos`, `1p`, `2p`).

***Experiment***: Train absolute encoding parser with [mGPT](https://huggingface.co/ai-forever/mGPT) as encoder and LSTM layer as decoder to predict labels.  

```shell
python3 -u -m supar.cmds.dep.sl train -b -c configs/config-mgpt.ini \
    -p ../results/models-dep/english-ewt/abs-mgpt-lstm/parser.pt \
    --codes abs --decoder lstm \
    --train ../treebanks/english-ewt/train.conllu \
    --dev ../treebanks/english-ewt/dev.conllu \
    --test ../treebanks/english-ewt/test.conllu
```
Model configuration (number and size of layers, optimization parameters, encoder selection) is specified using configuration files (see folder `configs/`). We provided the main configuration used for our experiments. 

* **Transition-based Dependency Parser w. Arc-Eager** ([`ArcEagerDependencyParser`](supar/models/dep/eager/parser.py)): Inherits the same arguments as the main class [`Parser`](supar/parser.py).

***Experiment***: Train Arc-Eager parser using [BLOOM-560M ](https://huggingface.co/bigscience/bloom-560m) as encoder and a MLP-based decoder to predict transitions with delay $k=1$ ( `--delay`) and Vector Quantization (`--use_vq`).

```shell
python3 -u -m supar.cmds.dep.eager train -b -c configs/config-bloom560.ini \
    -p ../results/models-dep/english-ewt/eager-bloom560-mlp/parser.pt \
    --decoder=mlp --delay=1 --use_vq \
    --train ../treebanks/english-ewt/train.conllu \
    --dev ../treebanks/english-ewt/dev.conllu \
    --test ../treebanks/english-ewt/test.conllu
```

This will save in folder `results/models-dep/english-ewt/eager-bloom560-mlp` the following files:

1. `parser.pt`: PyTorch trained model.
2. `metrics.pickle`: Python object with the evaluation of test set.
3. `pred.conllu`: Parser prediction of CoNLL-U test file.


**Constituency Parsing** 

* **Sequence Labeling Constituency Parser** ([`SLConstituencyParser`](supar/models/const/sl/parser.py)): Analogously to [`SLDependencyParser`](supar/models/dep/sl/parser.py), it allows the flag `--codes` in order to specify the indexing to use (`abs`, `rel`).

```shell 
python3 -u -m supar.cmds.const.sl train -b -c configs/config-mgpt.ini \
    -p ../results/models-con/ptb/abs-mgpt-lstm/parser.pt \
    --codes abs --decoder lstm \
    --train ../treebanks/ptb-gold/train.trees \
    --dev ../treebanks/ptb-gold/dev.trees \
    --test ../treebanks/ptb-gold/test.trees
```

* **Attach-Juxtapose Constituency Parser** ([`AttachJuxtaposeConstituencyParser`](supar/models/const/aj/parser.py)): From the original SuPar implementation, we added the delay and Vector Quantization flag:

```shell 
python3 -u -m supar.cmds.const.aj train -b -c configs/config-bloom560.ini \
    -p ../results/models-con/ptb/aj-bloom560-mlp/parser.pt \
    --delay=2 --use_vq \
    --train ../treebanks/ptb-gold/train.trees \
    --dev ../treebanks/ptb-gold/dev.trees \
    --test ../treebanks/ptb-gold/test.trees
```


### Evaluation

Our codes provides two evaluation methods from a `.pt` PyTorch:

1. Via Python prompt, loading the model with `.load()` method and evaluating with `.evaluate()`:

```py
>>> Parser.load('../results/models-dep/english-ewt/abs-mgpt-lstm/paser.pt').evaluate('../data/english-ewt/test.conllu')
```

2. Via terminal commands:

```shell
python -u -m supar.cmds.dep.sl evaluate -p --data ../data/english-ewt/test.conllu
```
### Prediction

Prediction step can be also executed from Python prompt or terminal commands to generate a CoNLL-U file:

1. Python terminal with `.predict()` method:

```py
>>> Parser.load('../results/models-dep/english-ewt/abs-mgpt-lstm/parser.pt')
    .predict(data='../data/english-ewt/abs-mgpt-lstm/test.conllu', 
            pred='../results/models-dep/english-ewt/abs-mgpt-lstm/pred.conllu')
```

2. Via terminal commands:
```shell 
python -u -m supar.cmds.dep.sl predict -p \ 
    --data ../data/english-ewt/test.conllu \
    --pred ../results/models-dep/english-ewt/abs-mgpt-lstm/pred.conllu
```

## Acknowledgments 

This work has been funded by the European Research Council (ERC), under the Horizon Europe research and innovation programme (SALSA, grant agreement No 101100615), ERDF/MICINN-AEI (SCANNER-UDC, PID2020-113230RB-C21), Xunta de Galicia (ED431C 2020/11), Cátedra CICAS (Sngular, University of A Coruña), and Centro de Investigación de Galicia ‘‘CITIC’’.

## Citation

```bib
@thesis{ezquerro-2023-syntactic,
  title     = {{Análisis sintáctico totalmente incremental basado en redes neuronales}},
  author    = {Ezquerro, Ana and Gómez-Rodríguez, Carlos and Vilares, David},
  institution = {University of A Coruña},
  year      = {2023},
  url       = {https://ruc.udc.es/dspace/handle/2183/33269}
}

@inproceedings{ezquerro-2023-challenges,
  title     = {{On the Challenges of Fully Incremental Neural Dependency Parsing}},
  author    = {Ezquerro, Ana and Gómez-Rodríguez, Carlos and Vilares, David},
  booktitle = {Proceedings of ICNLP-AACL 2023},
  year      = {2023}
}
```