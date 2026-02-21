#!/bin/bash
# Installation script for git hooks

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Installing git hooks for Bayesian Data Analysis project...${NC}"
echo ""

# Check if .git directory exists
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "Error: Not a git repository!"
    exit 1
fi

# Install pre-commit hook
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    echo -e "${YELLOW}⚠️  Pre-commit hook already exists!${NC}"
    echo "Backing up existing hook to pre-commit.backup"
    cp "$HOOKS_DIR/pre-commit" "$HOOKS_DIR/pre-commit.backup"
fi

echo "Installing pre-commit hook..."
cp "$SCRIPT_DIR/pre-commit.template" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo -e "${GREEN}✓ Pre-commit hook installed${NC}"
echo ""
echo "The hook will:"
echo "  - Remind you to update documentation when committing notebooks"
echo "  - Warn about large files (>1MB)"
echo "  - Suggest good commit message patterns"
echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "To disable temporarily: git commit --no-verify"
echo "To uninstall: rm .git/hooks/pre-commit"
