#!/usr/bin/env bash
# deploy_to_hf.sh — push the llm-judge app to a Hugging Face Space
#
# Usage (from the llm-judge directory):
#   bash scripts/deploy_to_hf.sh <hf-username> <space-name>
#
# Example:
#   bash scripts/deploy_to_hf.sh satishThakur llm-judge-bayes
#
# Prerequisites:
#   1. huggingface-hub installed:  pip install huggingface-hub
#   2. Logged in to HF:            huggingface-cli login
#   3. Space already created at:   https://huggingface.co/spaces/<username>/<space-name>
#      (Create it on the HF website — choose Streamlit SDK, any visibility)

set -e

HF_USER=${1:?"Usage: $0 <hf-username> <space-name>"}
SPACE_NAME=${2:?"Usage: $0 <hf-username> <space-name>"}
SPACE_REPO="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
TMP_DIR=$(mktemp -d)

echo "Deploying to: ${SPACE_REPO}"
echo "Temp dir: ${TMP_DIR}"
echo ""

# ── Clone the HF Space repo ────────────────────────────────────────────────────
echo "1. Cloning HF Space repo..."
git clone "https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}" "${TMP_DIR}"

# ── Copy app files ─────────────────────────────────────────────────────────────
echo "2. Copying app files..."
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cp "${SCRIPT_DIR}/app.py"           "${TMP_DIR}/app.py"
cp "${SCRIPT_DIR}/requirements.txt" "${TMP_DIR}/requirements.txt"
cp "${SCRIPT_DIR}/Dockerfile"       "${TMP_DIR}/Dockerfile"
cp "${SCRIPT_DIR}/HF_README.md"     "${TMP_DIR}/README.md"   # HF uses README.md for metadata

cp -r "${SCRIPT_DIR}/llm_judge"    "${TMP_DIR}/llm_judge"
cp -r "${SCRIPT_DIR}/pages"        "${TMP_DIR}/pages"
cp -r "${SCRIPT_DIR}/data"         "${TMP_DIR}/data"         # includes .nc files

# ── Git LFS for .nc files (HF recommends LFS for files > 10MB, ours are ~1MB each) ──
# Optional — uncomment if HF complains about file sizes
# cd "${TMP_DIR}" && git lfs install && git lfs track "*.nc"

# ── Commit and push ────────────────────────────────────────────────────────────
echo "3. Committing and pushing..."
cd "${TMP_DIR}"
git add .
git commit -m "Deploy llm-judge Bayesian inference app"
git push

echo ""
echo "✅ Deployed! View your Space at:"
echo "   ${SPACE_REPO}"
echo ""
echo "Note: HF Spaces takes 1-3 minutes to build after the push."
rm -rf "${TMP_DIR}"
