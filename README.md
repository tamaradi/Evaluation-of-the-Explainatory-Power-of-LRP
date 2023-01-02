# Evaluation of the Explainatory Power of LRP
This repository provides source code implemented in Python 3.8 to evaluate the explainatory power of layer-wise relevance propagation (LRP) - a saliency method for visualizing and explaining the decision process of convolutional neural networks (CNNs) - using adversarial examples. Therefore, it contains the following items:

* ![Simple CNN architecture](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/tree/main/src/model) (incl. ![training possibilities](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/blob/main/src/01_train_eval_CNN.py)), as well as the weights of a model trained on the benchmarkset CIFAR-10
* ![Funtionalities to generate adversarial examples](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/blob/main/src/02_generate_adversarial_examples.py) using the L-BFGS attack (build on ![Foolbox 2.4.0](https://github.com/bethgelab/foolbox/tree/v2))
* Implementation of the ![fundamental principles of LRP](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/tree/main/src/LRP) and ![a process to generate relevance scores](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/blob/main/src/03_run_LRP.py) for adversarial examples and original (correctly classified) images applying LRP
* Diverse evaluation scripts (e.g., ![visual verification via heatmaps](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/blob/main/src/04_create_exemplary_heatmaps.py), ![relevance ranking](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/blob/main/src/05_create_relevance_ranking.py))

The repository also includes exemplarily ![adversarial exampamples](https://github.com/tamaradi/Evaluation-of-the-Explainatory-Power-of-LRP/tree/main/data/adversarial_examples/Test) generated based on the test set of the benchmark dataset ![CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) using the L-BFGS attack. The adversarial attack was conducted for different target classes: 

* Target Class 0: airplane 										
* Target Class 1: automobile 										
* Target Class 2: bird 										
* Target Class 3: cat 										
* Target Class 4: deer 										
* Target Class 5: dog 										
* Target Class 6: frog 										
* Target Class 7: horse 										
* Target Class 8: ship 										
* Target Class 9: truck

## Citation
If you use funtionalities and scripts included in this repository, please cite our paper:
```
@article{dieter2023EvaluationOfLRP,
  title={Evaluation of the Explanatory Power Of Layer-wise Relevance Propagation using Adversarial Examples},
  author={Dieter, Tamara R. and Zisgen, Horst},
  journal={Neural Processing Letters},
  year={2023}
}
```
You can find our paper ....

## Autors
* Tamara R. Dieter
* Horst Zisgen

[![DOI](https://zenodo.org/badge/583320941.svg)](https://zenodo.org/badge/latestdoi/583320941)
