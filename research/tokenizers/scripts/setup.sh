#!/bin/bash
# Concept Tokenizer Database Setup Script
# Run this on frankenputer to set up the full pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
REFERENCE_DIR="$BASE_DIR/reference"
DB_DIR="$BASE_DIR/db"
VENV_DIR="$BASE_DIR/venv"

echo "=========================================="
echo "Concept Tokenizer DB Setup"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# 1. Create directory structure
# ==========================================
echo -e "${YELLOW}[1/6] Creating directory structure...${NC}"

mkdir -p "$REFERENCE_DIR"/{glottolog,unimorph,morpholex,grambank,wals}
mkdir -p "$DB_DIR"
mkdir -p "$SCRIPT_DIR"

echo -e "${GREEN}  ✓ Directories created${NC}"

# ==========================================
# 2. Set up Python virtual environment
# ==========================================
echo -e "${YELLOW}[2/6] Setting up Python virtual environment...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}  ✓ Virtual environment already exists${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# ==========================================
# 3. Install Python dependencies
# ==========================================
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"

pip install --quiet --upgrade pip

pip install --quiet \
    requests>=2.28.0 \
    tqdm>=4.64.0 \
    openpyxl>=3.1.0 \
    SPARQLWrapper>=2.0.0

echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# ==========================================
# 4. Download data sources
# ==========================================
echo -e "${YELLOW}[4/6] Downloading data sources...${NC}"
echo "  This may take a while depending on connection speed..."
echo ""

# Run the Python download script
python3 "$SCRIPT_DIR/download_sources.py" --base-dir "$REFERENCE_DIR"

echo -e "${GREEN}  ✓ Downloads complete${NC}"

# ==========================================
# 5. Apply database schema
# ==========================================
echo -e "${YELLOW}[5/6] Applying database schema...${NC}"

# Check if schema needs to be applied
if [ -f "$DB_DIR/tokenizer.db" ]; then
    # Check if secondary schema tables exist
    TABLE_COUNT=$(sqlite3 "$DB_DIR/tokenizer.db" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='language_families';")

    if [ "$TABLE_COUNT" -eq "0" ]; then
        echo "  Applying secondary schema..."
        sqlite3 "$DB_DIR/tokenizer.db" < "$DB_DIR/SCHEMA-linguistic-secondary.sql"
        echo -e "${GREEN}  ✓ Schema applied${NC}"
    else
        echo -e "${GREEN}  ✓ Schema already applied${NC}"
    fi
else
    echo -e "${RED}  ✗ tokenizer.db not found - run init_morphemes.py first${NC}"
    exit 1
fi

# ==========================================
# 6. Run import scripts
# ==========================================
echo -e "${YELLOW}[6/6] Running import scripts...${NC}"

# Import order matters!
echo "  Importing Glottolog (language hierarchy)..."
python3 "$SCRIPT_DIR/import_glottolog.py"

echo "  Importing WALS (typological features)..."
python3 "$SCRIPT_DIR/import_wals.py"

echo "  Importing Grambank (grammar features)..."
python3 "$SCRIPT_DIR/import_grambank.py"

echo "  Importing MorphoLex-en (English morphemes)..."
python3 "$SCRIPT_DIR/import_morpholex.py"

echo "  Importing UniMorph (inflection paradigms)..."
python3 "$SCRIPT_DIR/import_unimorph.py"

echo -e "${GREEN}  ✓ Imports complete${NC}"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "Database: $DB_DIR/tokenizer.db"
echo "Reference data: $REFERENCE_DIR/"
echo ""
echo "To activate the environment later:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To regenerate working DB with specific languages:"
echo "  python3 $SCRIPT_DIR/generate_working_db.py --config config.json"
echo ""

# Validation
echo "Quick validation:"
sqlite3 "$DB_DIR/tokenizer.db" "SELECT 'Languages:', COUNT(*) FROM language_families WHERE level='language';"
sqlite3 "$DB_DIR/tokenizer.db" "SELECT 'Surface forms:', COUNT(*) FROM surface_forms;"
sqlite3 "$DB_DIR/tokenizer.db" "SELECT 'Grammar rules:', COUNT(*) FROM grammar_rules;"
