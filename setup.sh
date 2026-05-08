#!/usr/bin/env bash
# DeepSeek-R1 W4A16 INT4 — Runtime Setup
# Usage: bash setup.sh
# Before running: download the weight file and compiled runtime from
# https://github.com/Wierzbowski-Alien/deepseek-r1-w4a16-demo/releases
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
echo " DeepSeek-R1 W4A16 INT4 — Setup"
echo " $(date)"
echo " Dossier : $ROOT"
echo "=============================================="
echo ""

# --- Check GPU ---
info "Vérification GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi introuvable. Une carte NVIDIA est requise."
    exit 1
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
ok "GPU détecté : $GPU ($VRAM)"

# --- Check for required binary files ---
REPO_URL="https://github.com/Wierzbowski-Alien/deepseek-r1-w4a16-demo"
RELEASES_URL="$REPO_URL/releases"

MISSING=""

if [ ! -f "$ROOT/deepseek_r1_w4a16_bi.pt" ]; then
    MISSING="$MISSING  - deepseek_r1_w4a16_bi.pt (5.3 GB, poids du modèle INT4)\n"
fi

SO_FILE=$(ls "$ROOT"/deepseek_r1_w4a16_C*.so 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    MISSING="$MISSING  - deepseek_r1_w4a16_C.*.so (14 MB, runtime CUDA compilé)\n"
fi

if [ -n "$MISSING" ]; then
    echo ""
    warn "Fichiers binaires manquants :"
    echo -e "$MISSING"
    warn "Ces fichiers ne sont PAS dans le dépôt git (trop volumineux)."
    warn "Téléchargez-les depuis les Releases GitHub :"
    echo ""
    echo "    $RELEASES_URL"
    echo ""
    warn "Placez les deux fichiers à la racine de ce dossier, puis relancez :"
    echo ""
    echo "    bash setup.sh"
    echo ""
    exit 1
fi
ok "Poids du modèle trouvés (5.3 GB)"
ok "Runtime CUDA trouvé ($(basename "$SO_FILE"))"

# --- Python & venv ---
if [ ! -d "$ROOT/venv" ]; then
    info "Création de l'environnement virtuel Python..."
    python3 -m venv "$ROOT/venv"
    ok "venv créé"
fi
source "$ROOT/venv/bin/activate"
ok "venv activé"

# --- Dépendances ---
info "Installation de PyTorch + CUDA 12.4..."
pip install --upgrade pip --quiet 2>/dev/null
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
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
