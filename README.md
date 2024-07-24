# Deep Jointly Optical Spectral Band Selection and Classification Learning

## Autors: Karen Fonseca, Jorge Bacca, Hans Garcia and Henry Arguello
## Universidad Industrial de Santander, Bucaramanga, Colombia

## Abstract

Spectral data provides material-specific information across a broad electromagnetic wavelength range by acquiring numerous spectral bands. However, acquiring such a significant volume of data introduces challenges such as data redundancy, long acquisition times, and substantial storage capacity. To address these challenges, band selection is introduced as a strategy that focuses on only using the most significant bands to preserve spectral information for a specific task. State-of-the-art methods focus on searching for the most significant bands from previously acquired data, regardless of the optical system and the classification model.
Nevertheless, some deep learning methods, such as end-to-end frameworks, allow the design of optical systems and the learning of the classification network parameters. In this paper, we model the optical band selection as a trainable layer that is coupled with a classification network, and the parameters are learned in an end-to-end framework. To guarantee a physically implementable system, we proposed two regularization terms in the training step to promote binarization and also the number of the selected bands, as we need to provide the conditions to design the physical element where the light passes through. The proposed method provides better performance than state-of-the-art band selection methods for three different spectral datasets under the same conditions.


## How to cite
If this code is useful for you and you use it in academic work, please consider citing this paper as:

Karen Fonseca, Jorge Bacca, Hans Garcia, and Henry Arguello, "Deep jointly optical spectral band selection and classification learning," Appl. Opt. 63, 5505-5514 (2024)

@article{Fonseca:24,
author = {Karen Fonseca and Jorge Bacca and Hans Garcia and Henry Arguello},
journal = {Appl. Opt.},
keywords = {Machine vision; Neural networks; Optical computing; Optical systems; Point spread function; Systems design},
number = {21},
pages = {5505--5514},
publisher = {Optica Publishing Group},
title = {Deep jointly optical spectral band selection and classification learning},
volume = {63},
month = {Jul},
year = {2024},
url = {https://opg.optica.org/ao/abstract.cfm?URI=ao-63-21-5505},
doi = {10.1364/AO.523199},
}
