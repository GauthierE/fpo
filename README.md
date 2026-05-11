# fpo

This repository contains the code for reproducing the experiments and figures presented in the paper [Explaining and Preventing Alignment Collapse in Iterative RLHF](https://arxiv.org/abs/2605.04266).

## Organization

The repository is structured into two main folders, each corresponding to one of the two experimental settings presented in the paper: controlled simulations on synthetic environments (`simulations`) and the LLM alignment pipeline using Llama-3.2-1B (`llm`). Each folder is self-contained and independent of the other.

## Instructions

### Simulations

The `simulations/` folder contains the two controlled environments used in the paper.

#### Linear environment

1. Open the `simulations/linear/linear.ipynb` notebook.
2. Run all cells to reproduce the experiments and generate the figures. The output figures (`noise.pdf`, `phase.pdf`, `utility.pdf`) will be saved in the `fig/practical/` and `fig/relaxed/` subfolders, corresponding to the two FPO variants studied in the paper.

#### Neural-network environment

1. Open the `simulations/neural-network/` folder and run the notebook to reproduce the experiments and generate the corresponding figures.

### LLM alignment pipeline

The `llm/` folder contains the code for the Llama-3.2-1B alignment experiment on TruthfulQA. The full pipeline (training, generation, evaluation) is driven by `fpo.py`.

1. Set up API credentials. A Hugging Face token is required to download Llama-3.2-1B, and a Together AI key is required for the LLM-as-a-judge evaluation:

   ```bash
   # Linux / macOS
   export HF_TOKEN="your_hf_key"
   export TOGETHER_API_KEY="your_together_api_key"
   ```

   ```powershell
   # Windows (PowerShell)
   $env:HF_TOKEN="your_hf_key"
   $env:TOGETHER_API_KEY="your_together_api_key"
   ```

2. (*Optional*) Run a dry-run sanity check before launching the full pipeline:

   ```bash
   python fpo.py --mode all --dry-run
   ```

3. Train the three policies (baseline, practical FPO, relaxed FPO). This is the longest step and incurs no API cost. The resulting LoRA weights are saved in `llama_fpo_baseline_weights/`, `llama_fpo_practical_weights/`, and `llama_fpo_relaxed_weights/`:

   ```bash
   python fpo.py --mode train
   ```

4. Generate model outputs on the TruthfulQA prompts. The generations are saved in `truthfulqa_generations.json` and no API call is made at this step:

   ```bash
   python fpo.py --mode generate
   ```

5. Run the pairwise LLM-as-a-judge evaluation. This step uses the Together AI API and produces `fpo_final_results_3way.csv`:

   ```bash
   python fpo.py --mode evaluate
   ```

   Alternatively, steps 3–5 can be run end-to-end with `python fpo.py --mode all`.

6. Print the final pairwise win-rate tables and verbosity check:

   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('fpo_final_results_3way.csv')
   for left, right, col in [('BASELINE','PRACTICAL','Win_Base_v_Prac'),
                            ('BASELINE','RELAXED','Win_Base_v_Rel'),
                            ('RELAXED','PRACTICAL','Win_Rel_v_Prac')]:
       print(f'\n--- {left} VS {right} ---')
       print(pd.crosstab(df['Category'], df[col], margins=True, margins_name='Total').to_string())
   print('\nVerbosity check:')
   print('Baseline:  {:.1f} words'.format(df['Len_Base'].mean()))
   print('Practical: {:.1f} words'.format(df['Len_Prac'].mean()))
   print('Relaxed:   {:.1f} words'.format(df['Len_Rel'].mean()))
   "
   ```

7. (*Optional*) Reproduce the MMLU and ARC-Challenge capability evaluations using [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness):

   ```bash
   pip install lm-eval[hf]

   for variant in baseline practical relaxed; do
     lm_eval \
       --model hf \
       --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,peft=./llama_fpo_${variant}_weights,dtype=bfloat16 \
       --tasks mmlu,arc_challenge \
       --device cuda \
       --batch_size 64 \
       --output_path ./eval_results_${variant}
   done
   ```

   Results are written to `eval_results_baseline/`, `eval_results_practical/`, and `eval_results_theoretical/`.
