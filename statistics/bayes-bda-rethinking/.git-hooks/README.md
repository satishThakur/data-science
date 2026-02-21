# Git Hooks for Documentation Maintenance

This directory contains git hook templates to help keep documentation up-to-date.

## Installation

### Quick Setup
```bash
# From repository root
bash .git-hooks/install.sh
```

### Manual Setup
```bash
# Copy the pre-commit hook
cp .git-hooks/pre-commit.template .git/hooks/pre-commit

# Make it executable
chmod +x .git/hooks/pre-commit
```

## What the Pre-Commit Hook Does

When you commit changes, the hook will:

1. **Detect notebook changes** - Check if any `.ipynb` files were modified
2. **Remind about documentation** - Suggest updating:
   - `TODO.md` - Progress tracking
   - `CLAUDE.md` - Project/chapter context
   - `MEMORY.md` - Learnings and patterns
3. **Check file sizes** - Warn about large files (>1MB)
4. **Suggest commit messages** - Show good examples

## Customization

Edit `.git-hooks/pre-commit.template` to:
- Change file size thresholds
- Add custom checks
- Modify reminder messages
- Add automatic formatting

## Disable Hook Temporarily

```bash
# Skip hooks for one commit
git commit --no-verify -m "your message"

# Disable hook permanently
rm .git/hooks/pre-commit
```

## Re-enable Hook

```bash
# Re-install from template
cp .git-hooks/pre-commit.template .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
