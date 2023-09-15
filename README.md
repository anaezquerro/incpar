# :pencil: [IncPar](https://github.com/anaezquerro/incpar): Fully [Inc](https://github.com/anaezquerro/incpar)remental Neural Dependency and Constituency [Par](https://github.com/anaezquerro/incpar)sing

A Python package for reproducing results of fully incremental dependency (comming soon constituency :raised_hands:) parsers described in ":page_facing_up:On The Challenges of Fully Incremental Neural Dependency Parsing" ([IJCNLP-AACL 2023](http://www.ijcnlp-aacl2023.org/)) and "[:page_facing_up:](https://ruc.udc.es/dspace/handle/2183/33269)Análisis sintáctico totalmente incremental basado en redes neuronales"  ([University of A Coruña](https://ruc.udc.es/)).

**Note**: Our implementation was built from forking [yzhangcs](https://github.com/yzhangcs)' [SuPar v1.1.4](https://github.com/yzhangcs/parser) repository. The Vector Quantization module was extracted from [lucidrains](https://github.com/lucidrains)' [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and Sequence Labeling encodings from [Polifack](https://github.com/Polifack)'s [CoDeLin](https://github.com/Polifack/codelin) repositories.

## Incremental Parsers

* Dependency Parsing:
    * [Sequence Labeling](https://aclanthology.org/N19-1077/) (absolute, relative, PoS-based and bracketing encodings).
    * [Transition-based w. Arc-Eager](https://aclanthology.org/C12-1059/).
* Constituency Parsing:
    * [Sequence Labeling](https://aclanthology.org/D18-1162/) (absolute and relative encodings).


## Usage

In order to reproduce our experiments, follow the installation and deployment steps of [SuPar](https://github.com/yzhangcs/parser), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) and [CoDeLin](https://github.com/Polifack/codelin) repositories.


