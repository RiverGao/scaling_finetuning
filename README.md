# Increasing Human-Model Alignment: Scaling vs. Finetuning

# System requirements
- OS: Ubuntu >= 20.04 or MacOS >= 14.6.1
- `torch`, `transformers`

# Installation guide
- Clone the repository with `git clone https://github.com/RiverGao/scaling_finetuning`

# Demo
1. The `utils` directory contains the python scripts to reproduce the results
2. The `expected_results` directory contains some of the expected outputs

# Instructions for use
1. (Optional) run `utils/get_attention/get_[model]_attention.py` to get the attention score of a given model
2. (Optional) modify and run `utils/get_residues/get_model_residues.sh` to get the residue of model attention scores
3. run `utils/divergence/do_llama_divergence.sh` to calculate the divergence of model attention scores
4. run `utils/regression/heads_vs_saccade.py` to perform regression between model attention and human saccade
5. run `utils/regression/heads_vs_fmri.py` to perform regression bewteen model attention and human fmri
