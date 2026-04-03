# IMO-Bench Reproduction

Standalone reproduction code for IMO-Bench evaluation, extracted from the larger RLVR workspace.

Scope:
- `IMO-AnswerBench`
- `IMO-ProofBench`
- `IMO-GradingBench`
- Gemini-based benchmark-faithful grader paths
- local Hugging Face judge paths for cheaper GradingBench calibration

Design policy:
- If the paper specifies a prompt or protocol detail, this repo follows it.
- If the paper does not specify a detail, this repo uses an explicit default and documents it as a house choice.
- Primary benchmark-faithful path:
  - `AnswerBench`: Gemini answer autograder
  - `ProofBench`: Gemini `ProofAutoGrader`-style prompt
  - `GradingBench`: vanilla grader prompt from the paper
- Optional cheaper path:
  - `AnswerBench`: local `Math-Verify` final-answer checker (`--answer-grader-backend math_verify`)

What this repo is for:
- reproducing IMO-Bench-style evaluation outside the RL training codebase
- running benchmark checks on a cluster or Colab
- comparing local open-source judge models on `GradingBench`

What this repo is not:
- a claim of exact score reproduction for Google-hosted models across time
- a replacement for human evaluation on `ProofBench`

## Files

- [eval_imobench.py](/mnt/home/bf2504/imobench-repro/eval_imobench.py): main evaluator
- [scripts/submit_imobench_eval.sbatch](/mnt/home/bf2504/imobench-repro/scripts/submit_imobench_eval.sbatch): Slurm submitter
- [requirements.txt](/mnt/home/bf2504/imobench-repro/requirements.txt): lightweight Python deps

## Install

On Colab or a fresh environment, PyTorch is often already present. Then install:

```bash
pip install -r requirements.txt
```

If you need PyTorch, install the CUDA-appropriate wheel first.

## Quick Start

### 1. Gemini smoke run

```bash
export GEMINI_API_KEY='...'
python3 eval_imobench.py \
  --benchmarks answerbench,proofbench,gradingbench \
  --limit-answerbench 2 \
  --limit-proofbench 2 \
  --limit-gradingbench 2 \
  --output-root ./imobench_runs/smoke
```

### 2. Paper-faithful full AnswerBench run with Gemini

```bash
export GEMINI_API_KEY='...'
python3 eval_imobench.py \
  --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --benchmarks answerbench \
  --gemini-model gemini-2.5-pro \
  --output-root ./imobench_runs/answerbench_gemini
```

### 3. Cheaper local AnswerBench run with Math-Verify

This is not the paper-faithful `AnswerAutoGrader`, but it can verify most
boxed final answers locally and includes a small fallback layer for the few
AnswerBench gold answers that are plain-text or tuple-style.

```bash
python3 eval_imobench.py \
  --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --benchmarks answerbench \
  --answer-grader-backend math_verify \
  --output-root ./imobench_runs/answerbench_math_verify
```

### 4. Paper-faithful GradingBench run with Gemini

```bash
export GEMINI_API_KEY='...'
python3 eval_imobench.py \
  --benchmarks gradingbench \
  --judge-backend gemini \
  --output-root ./imobench_runs/gradingbench_gemini
```

### 5. ProofBench run with concurrent Gemini judging

If you already have `proofbench_predictions.jsonl` and only need the judging
pass, the evaluator can resume from the existing predictions and issue several
Gemini grading requests in parallel.

```bash
export GEMINI_API_KEY='...'
python3 eval_imobench.py \
  --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B hbx/JustRL-DeepSeek-1.5B \
  --benchmarks proofbench \
  --gemini-model gemini-2.5-pro \
  --gemini-timeout 600 \
  --gemini-retries 5 \
  --judge-concurrency 4 \
  --output-root ./imobench_runs/proofbench_gemini_20260403
```

### 6. Local open-source judge comparison on one A100 80G

This is the intended Colab-friendly path. Start with `32B` judges, one at a time or sequentially in one job.

```bash
python3 eval_imobench.py \
  --benchmarks gradingbench \
  --judge-backend local_hf \
  --judge-models-csv "Qwen/QwQ-32B,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --gradingbench-per-point 10 \
  --gradingbench-seed 0 \
  --output-root ./imobench_runs/gradingbench_local32b
```

## Reproduction Notes

### ProofAutoGrader

The evaluator includes a `ProofAutoGrader`-style path based on the paper’s Appendix B.5 prompt structure:
- problem statement
- candidate solution
- reference solution
- specific grading guidelines

Remaining caveat:
- the prompt body is close to the paper, but hosted model behavior and some runtime details can still differ from the original authors’ setup
- judged `ProofBench` rows are checkpointed incrementally, so interrupted runs can resume without losing all completed Gemini grades from the current batch

### GradingBench

For `GradingBench`, the paper’s vanilla prompt uses:
- problem
- proposed solution

and predicts one of:
- `incorrect`
- `partial`
- `almost`
- `correct`

This repo reports:
- `accuracy`
- `mae_percent`

and also logs extra debugging metrics such as confusion matrices and catastrophic overgrade rate.

## Slurm

Submit with:

```bash
sbatch scripts/submit_imobench_eval.sbatch
```

Example local-HF GradingBench dev slice:

```bash
sbatch --export=ALL,JUDGE_BACKEND=local_hf,BENCHMARKS=gradingbench,GRADINGBENCH_PER_POINT=10,JUDGE_MODELS='Qwen/QwQ-32B,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',RUN_TAG=gradingbench-local32b scripts/submit_imobench_eval.sbatch
```
