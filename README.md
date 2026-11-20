# Passport Please

Passport Please: a 30-minute in-class Colab activity on nationality bias in three small models: text to text
(Qwen2.5-0.5B-Instruct), text to image (SD-Turbo) and image to text (SmolVLM-256M-Instruct).

1. **Smuggle the Stereotype.** Add one everyday detail to a fixed scene and see whether it moves
   the outputs to a given country.
2. **Customs Check.** Add as few words as possible to a skewed default prompt so the outputs spread
   across eight countries (USA, Mexico, Nigeria, Egypt, India, China, Greece, Germany).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chahatraj/Passport-Please/blob/main/passport_please.ipynb)

In Colab, choose Runtime → Change runtime type → T4 GPU.

`build_activity.py` writes the notebook; edit it, not the .ipynb.

Every run is recorded anonymously (a one-line notice at the top says so). Runs, with their images, go to a Google Sheet and Drive folder through a small Apps Script web app: see `apps_script/SETUP.md`. Nothing is sent until `LOG_URL` is set in `build_activity.py`.
