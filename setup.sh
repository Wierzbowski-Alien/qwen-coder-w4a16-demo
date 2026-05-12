#!/usr/bin/env bash
# Qwen 2.5 Coder W4A16 INT4 — Runtime Setup
# Usage: bash setup.sh
# Before running: download the weight file and compiled runtime from
# https://github.com/Wierzbowski-Alien/qwen-coder-w4a16-demo/releases
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()    { echo -e "${GREEN}[+]${NC} $1"; }
err()   { echo -e "${RED}[!]${NC} $1"; }
warn()  { echo -e "${YELLOW}[~]${NC} $1"; }

echo ""
echo "=============================================="
echo " Qwen 2.5 Coder W4A16 INT4 — Setup"
echo " $(date)"
echo " Dossier : $ROOT"
echo "=============================================="
echo ""

# --- Check GPU ---
info "Verification GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi introuvable. Une carte NVIDIA est requise."
    exit 1
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
ok "GPU detecte : $GPU ($VRAM)"

# --- Check for required binary files ---
REPO_URL="https://github.com/Wierzbowski-Alien/qwen-coder-w4a16-demo"
RELEASES_URL="$REPO_URL/releases"

WEIGHTS_FILE="qwen2.5_coder_7b_w4a16_bi.pt"
SO_PREFIX="deepseek_r1_w4a16_C"

MISSING=""

# --- Poids du modele (5.3 GB, split en 3 parties dans les Releases) ---
if [ ! -f "$ROOT/$WEIGHTS_FILE" ]; then
    PART1="$ROOT/${WEIGHTS_FILE}.partaa"
    PART2="$ROOT/${WEIGHTS_FILE}.partab"
    PART3="$ROOT/${WEIGHTS_FILE}.partac"
    if [ -f "$PART1" ] && [ -f "$PART2" ] && [ -f "$PART3" ]; then
        info "Reassemblage des poids depuis les 3 parties..."
        cat "$PART1" "$PART2" "$PART3" > "$ROOT/$WEIGHTS_FILE"
        ok "Poids reassembles (5.3 GB)"
        info "Suppression des parties..."
        rm "$PART1" "$PART2" "$PART3"
        ok "Parties supprimees — vous pouvez les garder pour redistribuer"
    else
        MISSING="$MISSING  - ${WEIGHTS_FILE}.partaa (2.0 GB)\n"
        MISSING="$MISSING  - ${WEIGHTS_FILE}.partab (2.0 GB)\n"
        MISSING="$MISSING  - ${WEIGHTS_FILE}.partac (1.4 GB)\n"
    fi
fi
if [ -f "$ROOT/$WEIGHTS_FILE" ]; then
    ok "Poids du modele trouves (5.3 GB)"
fi

# --- Runtime CUDA compile (15 MB) ---
SO_FILE=$(ls "$ROOT"/${SO_PREFIX}*.so 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    MISSING="$MISSING  - ${SO_PREFIX}.*.so (15 MB, runtime CUDA compile)\n"
else
    ok "Runtime CUDA trouve ($(basename "$SO_FILE"))"
fi

if [ -n "$MISSING" ]; then
    echo ""
    warn "Fichiers binaires manquants :"
    echo -e "$MISSING"
    warn "Ces fichiers ne sont PAS dans le dépôt git (trop volumineux)."
    warn "Téléchargez TOUS les fichiers depuis les Releases GitHub :"
    echo ""
    echo "    $RELEASES_URL"
    echo ""
    warn "Placez-les à la racine de ce dossier, puis relancez :"
    echo ""
    echo "    bash setup.sh"
    echo ""
    exit 1
fi

# --- Python & venv ---
if [ ! -d "$ROOT/venv" ]; then
    info "Création de l'environnement virtuel Python..."
    python3 -m venv "$ROOT/venv" --system-site-packages
    ok "venv créé"
fi
source "$ROOT/venv/bin/activate"
ok "venv activé"

# --- Dépendances ---
pip install --upgrade pip --quiet 2>/dev/null

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    ok "PyTorch $TORCH_VER déjà disponible (CUDA OK)"
else
    info "Installation de PyTorch (CUDA)..."
    pip install torch torchvision torchaudio --quiet
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        err "PyTorch installé mais CUDA indisponible. Vérifiez vos pilotes NVIDIA."
        exit 1
    fi
    ok "PyTorch installé"
fi

pip install transformers tokenizers sentencepiece flask --quiet
ok "Dépendances installées"

echo ""
echo "=============================================="
echo " Setup terminé !"
echo ""
echo " Lancement du serveur → http://localhost:8080"
echo " Ctrl+C pour arrêter"
echo "=============================================="
echo ""

# --- Lancement ---
python "$ROOT/server.py"
