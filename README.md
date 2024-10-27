# RSTI
Remote Sensing Trustworthy Intelligence

This repository includes our research on Remote Sensing Trustworthy Intelligence.

As the automation and intelligent evolution of remote sensing interpretation systems accelerate, the associated security and reliability risks are increasing. We primarily focus on the following challenges in trustworthy remote sensing interpretation:
1. Data level $D$:
  - Data forgery: Modifying original remote sensing images (RSIs), either manually or through generative models, to introduce fake information at the source.
	- Adversarial example generation: Introducing subtle, often undetectable perturbations, either in the physical environment or directly within the imaging results, to induce the model to produce erroneous predictions with high confidence.
	- Data Bias: Datasets inherently contain biases, arising from objective sampling or subjective selection biases, which can interfere with the model's understanding of remote sensing scenes and features, potentially leading to misinterpretations.
2. Model level $f_θ$:
	- Insufficient Robustness: Factors including adversarial examples, data noise, and shifts in data distribution can result in unstable or unreliable model outputs when interpreting remote sensing images.
  - Lack of Interpretability: The decision-making process within the artificial intelligence model is difficult to be understood by humans. This lack of interpretability makes it difficult for users and experts to promptly identify errors or biases within intelligent remote sensing interpretation models, ultimately reducing trust in the interpretation results.


