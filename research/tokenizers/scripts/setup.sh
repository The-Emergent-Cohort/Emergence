#!/bin/bash
#
# Tokenizer Database System - Full Setup
#
# Run from: /usr/share/databases/scripts/ (symlinked from project)
# Requires: Python venv at ../venv/
#
# This script orchestrates the full import pipeline:
#   Phase 1: Foundation (schemas, NSM, Glottolog, WALS, Grambank)
#   Phase 2: Extended primitives (tarballs, VerbNet, WordNet)
#   Phase 3: Multilingual (OMW)
#   Phase 4: Lexical data (Kaikki)
#   Phase 5: Analysis (compositions, master index)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$BASE_DIR/venv"
DB_DIR="$BASE_DIR/db"
REF_DIR="$BASE_DIR/reference"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_phase() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

echo_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

echo_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

echo_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

echo_skip() {
    echo -e "${YELLOW}[SKIP] $1${NC}"
}

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo_error "Virtual environment not found at $VENV_DIR"
    echo "Create it with: python3 -m venv $VENV_DIR"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

echo ""
echo "Tokenizer Database System - Full Setup"
echo "======================================="
echo "Script directory: $SCRIPT_DIR"
echo "Base directory:   $BASE_DIR"
echo "Database output:  $DB_DIR"
echo "Reference data:   $REF_DIR"
echo ""

# Parse arguments
PHASE="${1:-all}"
LANG="${2:-en}"

run_script() {
    local script="$1"
    local args="${2:-}"

    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo_step "Running $script $args"
        python "$SCRIPT_DIR/$script" $args
        echo_success "$script completed"
    else
        echo_skip "$script not found"
        return 1
    fi
}

check_reference() {
    local name="$1"
    local dir="$REF_DIR/$name"

    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return 0
    else
        echo_skip "Reference data not found: $dir"
        return 1
    fi
}

# Phase 1: Foundation
phase1() {
    echo_phase "Phase 1: Foundation"

    echo_step "Initializing database schemas..."
    run_script "init_schemas.py"

    echo_step "Importing NSM semantic primes..."
    run_script "import_nsm_primes.py"

    if check_reference "glottolog"; then
        run_script "import_glottolog.py"
    fi

    if check_reference "wals"; then
        run_script "import_wals.py"
    fi

    if check_reference "grambank"; then
        run_script "import_grambank.py"
    fi

    echo_success "Phase 1 complete"
}

# Phase 2: Extended Primitives
phase2() {
    echo_phase "Phase 2: Extended Primitives"

    # Unpack tarballs if script exists
    if [ -f "$SCRIPT_DIR/unpack_tarballs.py" ]; then
        run_script "unpack_tarballs.py"
    else
        echo_skip "unpack_tarballs.py not implemented yet"
    fi

    if check_reference "verbnet"; then
        run_script "import_verbnet.py" 2>/dev/null || echo_skip "import_verbnet.py not implemented yet"
    fi

    if check_reference "wordnet"; then
        run_script "import_wordnet.py" 2>/dev/null || echo_skip "import_wordnet.py not implemented yet"
    fi

    echo_success "Phase 2 complete"
}

# Phase 3: Multilingual
phase3() {
    echo_phase "Phase 3: Multilingual"

    if check_reference "omw"; then
        run_script "import_omw.py" 2>/dev/null || echo_skip "import_omw.py not implemented yet"
    fi

    echo_success "Phase 3 complete"
}

# Phase 4: Lexical Data
phase4() {
    echo_phase "Phase 4: Lexical Data ($LANG)"

    if check_reference "kaikki"; then
        if [ -f "$SCRIPT_DIR/import_kaikki.py" ]; then
            run_script "import_kaikki.py" "--lang $LANG"
        else
            echo_skip "import_kaikki.py not implemented yet"
        fi
    fi

    echo_success "Phase 4 complete"
}

# Phase 5: Analysis
phase5() {
    echo_phase "Phase 5: Analysis"

    if [ -f "$SCRIPT_DIR/compute_compositions.py" ]; then
        run_script "compute_compositions.py" "--lang $LANG"
    else
        echo_skip "compute_compositions.py not implemented yet"
    fi

    if [ -f "$SCRIPT_DIR/build_master_index.py" ]; then
        run_script "build_master_index.py"
    else
        echo_skip "build_master_index.py not implemented yet"
    fi

    echo_success "Phase 5 complete"
}

# Run phases
case "$PHASE" in
    1|phase1|foundation)
        phase1
        ;;
    2|phase2|primitives)
        phase2
        ;;
    3|phase3|multilingual)
        phase3
        ;;
    4|phase4|lexical)
        phase4
        ;;
    5|phase5|analysis)
        phase5
        ;;
    2-5|rest)
        phase2
        phase3
        phase4
        phase5
        ;;
    all)
        phase1
        phase2
        phase3
        phase4
        phase5
        ;;
    *)
        echo "Usage: $0 [phase] [lang]"
        echo ""
        echo "Phases:"
        echo "  1, phase1, foundation   - Init schemas, NSM, Glottolog, WALS, Grambank"
        echo "  2, phase2, primitives   - VerbNet, WordNet"
        echo "  3, phase3, multilingual - OMW"
        echo "  4, phase4, lexical      - Kaikki (default: en)"
        echo "  5, phase5, analysis     - Compositions, Master index"
        echo "  2-5, rest               - Skip Phase 1, run 2-5"
        echo "  all                     - Run all phases (default)"
        echo ""
        echo "Examples:"
        echo "  $0                      # Run all phases for English"
        echo "  $0 1                    # Run only Phase 1"
        echo "  $0 2-5 en               # Skip Phase 1, run rest for English"
        echo "  $0 4 de                 # Run Phase 4 for German"
        echo "  $0 all ja               # Run all phases for Japanese"
        exit 1
        ;;
esac

echo ""
echo_phase "Setup Complete"
echo ""
echo "Databases created in: $DB_DIR"
echo ""

# Show summary
if [ -f "$DB_DIR/primitives.db" ]; then
    echo "primitives.db:"
    sqlite3 "$DB_DIR/primitives.db" "SELECT '  Primitives: ' || COUNT(*) FROM primitives;"
    sqlite3 "$DB_DIR/primitives.db" "SELECT '  Forms: ' || COUNT(*) FROM primitive_forms;"
fi

if [ -f "$DB_DIR/language_registry.db" ]; then
    echo ""
    echo "language_registry.db:"
    sqlite3 "$DB_DIR/language_registry.db" "SELECT '  Languages: ' || COUNT(*) FROM language_codes;"
    sqlite3 "$DB_DIR/language_registry.db" "SELECT '  Features: ' || COUNT(*) FROM language_features;"
fi

if [ -d "$DB_DIR/lang" ]; then
    echo ""
    echo "Language databases:"
    for db in "$DB_DIR/lang"/*.db; do
        if [ -f "$db" ]; then
            name=$(basename "$db" .db)
            count=$(sqlite3 "$db" "SELECT COUNT(*) FROM concepts;" 2>/dev/null || echo "0")
            echo "  $name: $count concepts"
        fi
    done
fi

echo ""
deactivate
