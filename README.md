# RSTI
**R**emote **S**ensing **T**rustworthy **I**ntelligence

## Overview
This repository includes our research on Remote Sensing Trustworthy Intelligence.

As the automation and intelligent evolution of remote sensing interpretation systems accelerate, the associated security and reliability risks are increasing. We primarily focus on the following challenges in trustworthy remote sensing interpretation:
1. Data level $D$:
	- Data forgery: Modifying original remote sensing images (RSIs), either manually or through generative models, to introduce fake information at the source.
	- Adversarial example: Introducing subtle, often undetectable perturbations, either in the physical environment or directly within the imaging results, to induce the model to produce erroneous predictions with high confidence.
	- Data Bias: Datasets inherently contain biases, arising from objective sampling or subjective selection biases, which can interfere with the model's understanding of remote sensing scenes and features, potentially leading to misinterpretations.
2. Model level $f_θ$:
	- Insufficient Robustness: Factors including adversarial examples, data noise, and shifts in data distribution can result in unstable or unreliable model outputs when interpreting remote sensing images.
  	- Lack of Interpretability: The decision-making process within the artificial intelligence model is difficult to be understood by humans. This lack of interpretability makes it difficult for users and experts to promptly identify errors or biases within intelligent remote sensing interpretation models, ultimately reducing trust in the interpretation results.


To address the above issues, we have conducted the following works.

## List of our works
### 1. Review
- L. Chen, Z. Xu, Q. Li, J. Peng, S. Wang and H. Li, "An Empirical Study of Adversarial Examples on Remote Sensing Image Scene Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 9, pp. 7419-7433, Sept. 2021, doi: 10.1109/TGRS.2021.3051641.
RSI forgery detection
- To be added later...

### 2. Data level

#### 2.1 RSI forgery detection
- Ji Q, Xinchang Zhang, Dingqi Ye, Yongjian Ruan, Zeyi Li, Xin Guo, Shaowen Wang and Haifeng Li. 2024. "SFNet: Fusion of Spatial and Frequency-Domain Features for Remote Sensing Image Forgery Detection". IEEE Transactions on Geoscience and Remote Sensing. (Under review)

#### 2.2 Adversarial example recognition
- Li Chen, Qi Li, Weiye Chen, Zeyu Wang, and Haifeng Li. 2022. A data‐driven adversarial examples recognition framework via adversarial feature genomes. Int. J. Intell. Syst. 37, 9 (September 2022), 6438–6462. https://doi.org/10.1002/int.22850


### 3. Model level
#### 3.1 Defence methods against adversarial samples
- L. Chen, J. Xiao, P. Zou and H. Li, "Lie to Me: A Soft Threshold Defense Method for Adversarial Examples of Remote Sensing Images," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 8016905, doi: 10.1109/LGRS.2021.3096244.
- To be added later...
