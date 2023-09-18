# :pencil: [IncPar](https://github.com/anaezquerro/incpar): Fully [Inc](https://github.com/anaezquerro/incpar)remental Neural Dependency and Constituency [Par](https://github.com/anaezquerro/incpar)sing

A Python package for reproducing results of fully incremental dependency (comming soon constituency :raised_hands:) parsers described in ":page_facing_up:On The Challenges of Fully Incremental Neural Dependency Parsing" ([IJCNLP-AACL 2023](http://www.ijcnlp-aacl2023.org/)) and "[:page_facing_up:](https://ruc.udc.es/dspace/handle/2183/33269)Análisis sintáctico totalmente incremental basado en redes neuronales"  ([University of A Coruña](https://ruc.udc.es/)).

**Note**: Our implementation was built from forking [yzhangcs](https://github.com/yzhangcs)' [SuPar v1.1.4](https://github.com/yzhangcs/parser) repository. The Vector Quantization module was extracted from [lucidrains](https://github.com/lucidrains)' [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and Sequence Labeling encodings from [Polifack](https://github.com/Polifack)'s [CoDeLin](https://github.com/Polifack/codelin) repositories.

## Incremental Parsers

* Dependency Parsing:
    * [Sequence Labeling](https://aclanthology.org/N19-1077/) (absolute, relative, PoS-based and bracketing encodings).
    * [Transition-based w. Arc-Eager](https://aclanthology.org/C12-1059/).
* Constituency Parsing:
    * [Sequence Labeling](https://aclanthology.org/D18-1162/) (absolute and relative encodings).
    * [Attach-Juxtapose](https://arxiv.org/abs/2010.14568).


## Usage

In order to reproduce our experiments, follow the installation and deployment steps of [SuPar](https://github.com/yzhangcs/parser), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and [CoDeLin](https://github.com/Polifack/codelin) repositories. Supported functionalities are **training**, **evaluation** and **prediction** from CoNLL-U or PTB-bracketed files. We highly suggest to run our parsers using terminal commands in order to train and generate prediction files. In the future we'll make available [SuPar methods](https://github.com/yzhangcs/parser#usage) to easily test our parsers' performance from Python terminal.

### Training

**Dependency Parsing**:

* **Sequence labeling Dependency Parser** ([`SLDependencyParser`](supar/models/dep/sl/parser.py)): Inherits all arguments of the main class [`Parser`](supar/parser.py) and allows the flag `--sl_codes` to specify encoding to configure the trees linearization (`abs`, `rel`, `pos`, `1p`, `2p`).

***Experiment***: Train absolute encoding parser with [mGPT](https://huggingface.co/ai-forever/mGPT) as encoder and LSTM layer as decoder to predict labels.  

```shell
python3 -u -m supar.cmds.dep.sl train -b -c configs/config-mgpt.ini \ 
    -p ../results/models-dep/english-ewt/abs-mgpt-lstm/parser.pt \ 
    -- sl_codes=abs --decoder=lstm \  
    --train ../data/english-ewt/train.conllu \ 
    --dev ../data/english-ewt/dev.conllu \ 
    --test ../data/english-ewt/test.conllu \ 
    --save_eval=../results/models-dep/english-ewt/abs-mgpt-lstm/metrics.pickle \ 
    --save_predict=../results/models-dep/english-ewt/abs-mgpt-lstm/pred.conllu
```
Model configuration (number and size of layers, optimization parameters, encoder selection) is specified using configuration files (see folder `configs/`). We provided the main configuration used for our experiments. 

* **Transition-based Dependency Parser w. Arc-Eager** ([`ArcEagerdependencyParser`](supar/models/dep/eager/parser.py)): Inherits the same arguments as the main class [`Parser`](supar/parser.py).

***Experiment***: Train Arc-Eager parser using [BLOOM-560M ](https://huggingface.co/bigscience/bloom-560m) as encoder and a MLP-based decoder to predict transitions with delay $k=1$ ( `--delay`) and Vector Quantization (`--use_vq`).

```shell
python3 -u -m supar.cmds.dep.eager train -b -c configs/config-bloom560.ini \ 
    -p ../results/models-dep/english-ewt/eager-bloom560-mlp/parser.pt \ 
    --decoder=mlp --delay=1 --use_vq \  
    --train ../data/english-ewt/train.conllu \ 
    --dev ../data/english-ewt/dev.conllu \ 
    --test ../data/english-ewt/test.conllu \ 
    --save_eval=../results/models-dep/english-ewt/eager-bloom560-mlp/metrics.pickle \ 
    --save_predict=../results/models-dep/english-ewt/eager-bloom560-mlp/pred.conllu
```

This will save in folder `results/models-dep/english-ewt/eager-bloom560-mlp` the following files:

1. `parser.pt`: PyTorch trained model.
2. `metrics.pickle`: Python object with the evaluation of test set.
3. `pred.conllu`: Parser prediction of CoNLL-U test file.


[$\textcolor{magenta}{\textsf{comming soon}}$] **Constituency Parsing** 


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


