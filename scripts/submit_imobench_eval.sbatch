#!/bin/bash
#SBATCH -p columbia
#SBATCH -A columbia
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclude=alphagpu51,alphagpu52,alphagpu53,alphagpu54
#SBATCH --job-name=IMO-EVAL
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bf2504@columbia.edu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-bwf-verl}"
GEMINI_API_ENV="${GEMINI_API_ENV:-GEMINI_API_KEY}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-pro}"
JUDGE_BACKEND="${JUDGE_BACKEND:-gemini}"
JUDGE_MODELS="${JUDGE_MODELS:-Qwen/QwQ-32B,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
JUDGE_MAX_INPUT_LENGTH="${JUDGE_MAX_INPUT_LENGTH:-8192}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"
BENCHMARKS="${BENCHMARKS:-answerbench,proofbench,gradingbench}"
MODEL_A="${MODEL_A:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_B="${MODEL_B:-hbx/JustRL-DeepSeek-1.5B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-2048}"
PROOF_MAX_NEW_TOKENS="${PROOF_MAX_NEW_TOKENS:-8192}"
ANSWER_RUNS="${ANSWER_RUNS:-8}"
ANSWER_TEMPERATURE="${ANSWER_TEMPERATURE:-1.0}"
ANSWER_TOP_P="${ANSWER_TOP_P:-1.0}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
LIMIT_ANSWERBENCH="${LIMIT_ANSWERBENCH:-0}"
LIMIT_PROOFBENCH="${LIMIT_PROOFBENCH:-0}"
LIMIT_GRADINGBENCH="${LIMIT_GRADINGBENCH:-0}"
GRADINGBENCH_PER_POINT="${GRADINGBENCH_PER_POINT:-0}"
GRADINGBENCH_SEED="${GRADINGBENCH_SEED:-0}"
IMOBENCH_DATA_DIR="${IMOBENCH_DATA_DIR:-/mnt/lustre/columbia/bf2504/data/imobench}"
RUN_TAG="${RUN_TAG:-imobench-$(date +%Y%m%d_%H%M%S)-job${SLURM_JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/lustre/columbia/bf2504/data/imobench_eval/${RUN_TAG}}"

if [ "${JUDGE_BACKEND}" = "gemini" ] && [ -z "${!GEMINI_API_ENV:-}" ]; then
    echo "Missing Gemini API key env var: ${GEMINI_API_ENV}"
    echo "Export ${GEMINI_API_ENV} before submitting."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
mkdir -p "${REPO_ROOT}/logs"

cd "${REPO_ROOT}"

echo "============================================"
echo "IMO-Bench Offline Evaluation"
echo "============================================"
echo "Conda Env: ${CONDA_ENV}"
echo "Benchmarks: ${BENCHMARKS}"
echo "Judge Backend: ${JUDGE_BACKEND}"
echo "Gemini Model: ${GEMINI_MODEL}"
echo "Judge Models: ${JUDGE_MODELS}"
echo "Model A: ${MODEL_A}"
echo "Model B: ${MODEL_B}"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Data Dir: ${IMOBENCH_DATA_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Answer Max New Tokens: ${ANSWER_MAX_NEW_TOKENS}"
echo "Proof Max New Tokens: ${PROOF_MAX_NEW_TOKENS}"
echo "Answer Runs: ${ANSWER_RUNS}"
echo "Answer Temperature: ${ANSWER_TEMPERATURE}"
echo "Answer Top-p: ${ANSWER_TOP_P}"
echo "Judge Temperature: ${JUDGE_TEMPERATURE}"
echo "Judge Top-p: ${JUDGE_TOP_P}"
echo "Judge Max Input Length: ${JUDGE_MAX_INPUT_LENGTH}"
echo "Judge Max New Tokens: ${JUDGE_MAX_NEW_TOKENS}"
echo "Limit AnswerBench: ${LIMIT_ANSWERBENCH}"
echo "Limit ProofBench: ${LIMIT_PROOFBENCH}"
echo "Limit GradingBench: ${LIMIT_GRADINGBENCH}"
echo "GradingBench Per-Point: ${GRADINGBENCH_PER_POINT}"
echo "GradingBench Seed: ${GRADINGBENCH_SEED}"
echo "============================================"

python3 "${REPO_ROOT}/eval_imobench.py" \
    --models "${MODEL_A}" "${MODEL_B}" \
    --benchmarks "${BENCHMARKS}" \
    --data-dir "${IMOBENCH_DATA_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --batch-size "${BATCH_SIZE}" \
    --answer-max-new-tokens "${ANSWER_MAX_NEW_TOKENS}" \
    --proof-max-new-tokens "${PROOF_MAX_NEW_TOKENS}" \
    --answer-runs "${ANSWER_RUNS}" \
    --answer-temperature "${ANSWER_TEMPERATURE}" \
    --answer-top-p "${ANSWER_TOP_P}" \
    --limit-answerbench "${LIMIT_ANSWERBENCH}" \
    --limit-proofbench "${LIMIT_PROOFBENCH}" \
    --limit-gradingbench "${LIMIT_GRADINGBENCH}" \
    --gradingbench-per-point "${GRADINGBENCH_PER_POINT}" \
    --gradingbench-seed "${GRADINGBENCH_SEED}" \
    --judge-backend "${JUDGE_BACKEND}" \
    --judge-models-csv "${JUDGE_MODELS}" \
    --gemini-model "${GEMINI_MODEL}" \
    --gemini-api-env "${GEMINI_API_ENV}" \
    --judge-max-input-length "${JUDGE_MAX_INPUT_LENGTH}" \
    --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS}" \
    --judge-temperature "${JUDGE_TEMPERATURE}" \
    --judge-top-p "${JUDGE_TOP_P}"
