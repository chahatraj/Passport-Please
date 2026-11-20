# Smuggle the stereotype, then debias golf

A 30-minute in-class Colab activity on nationality bias in three small models: text to text
(Qwen2.5-0.5B-Instruct), text to image (SD-Turbo) and image to text (SmolVLM-256M-Instruct).

1. **Smuggle the stereotype.** Add one everyday detail to a fixed scene and see whether it moves
   the outputs to a given country.
2. **Debias golf.** Add as few words as possible to a skewed default prompt so the outputs spread
   across eight countries (USA, Mexico, Nigeria, Egypt, India, China, Greece, Germany).

Open in Colab (Runtime → T4 GPU): `smuggle_and_golf.ipynb`.

`build_activity.py` writes the notebook; edit it, not the .ipynb.

Sharing answers for research is opt-in (one checkbox at the top) and off unless `LOG_URL` is set.
