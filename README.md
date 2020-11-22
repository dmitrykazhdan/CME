# CME: A Concept-based Model Extraction Framework

This repository contains an implementation of CME.
CME is a (C)oncept-based (M)odel(E)xtraction framework, which can be used for analysing 
DNN models via explainable concept-based extracted models, in order to explain and improve 
performance of DNN models, as well as to extract useful knowledge from them.
 
For further details, see [our paper](https://arxiv.org/abs/2010.13233).

The experiments use the following open-source datasets:

- [dSprites](https://github.com/deepmind/dsprites-dataset)
- [Caltech-UCSD Birds 200 (CUB)](http://www.vision.caltech.edu/visipedia/CUB-200.html)


Abstract
---

Deep Neural Networks (DNNs) have achieved remarkable performance on a range of tasks. 
A key step to further empowering DNN-based approaches is improving their explainability. 
In this work we present CME: a concept-based model extraction framework, 
used for analysing DNN models via concept-based extracted models. 
Using two case studies (dSprites, and Caltech UCSD Birds), 
we demonstrate how CME can be used to (i) analyse the concept information learned by a 
DNN model (ii) analyse how a DNN uses this concept information when predicting output labels 
(iii) identify key concept information that can further improve DNN predictive performance 
(for one of the case studies, we showed how model accuracy can be improved by over 14%, 
using only 30% of the available concepts). 


![alt text](https://github.com/dmitrykazhdan/CME/blob/master/figures/vis_abs_1.png)

![alt text](https://github.com/dmitrykazhdan/CME/blob/master/figures/vis_abs_2.png)


Prerequisites
---
TBC...

Citing
---

If you find this code useful in your research, please consider citing:

```
@article{kazhdan2020now,
  title={Now You See Me (CME): Concept-based Model Extraction},
  author={Kazhdan, Dmitry and Dimanov, Botty and Jamnik, Mateja and Li{\`o}, Pietro and Weller, Adrian},
  journal={arXiv preprint arXiv:2010.13233},
  year={2020}
}
```

