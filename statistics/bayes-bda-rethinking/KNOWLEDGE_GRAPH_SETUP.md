# Knowledge Graph Setup - Complete Documentation System

This document describes the comprehensive knowledge graph and documentation structure for the Bayesian Data Analysis project.

## Overview

We've created a multi-layered documentation system to:
- **Reduce token usage** in Claude Code sessions
- **Preserve context** across sessions
- **Track progress** systematically
- **Capture learnings** and patterns

## Documentation Structure

```
bayes-bda-rethinking/
├── CLAUDE.md                    # Root project context (MAIN ENTRY POINT)
├── TODO.md                      # Progress tracking & next steps
├── KNOWLEDGE_GRAPH_SETUP.md     # This file - system overview
├── notebooks/
│   └── chapter6/
│       ├── CLAUDE.md            # Chapter-specific context
│       ├── multicollinearity.ipynb
│       ├── collider_bias.ipynb
│       └── ...
├── .git-hooks/                  # Git hooks for doc maintenance
│   ├── README.md
│   ├── install.sh
│   └── pre-commit.template
└── ~/.claude/projects/.../memory/
    └── MEMORY.md                # Persistent learnings (auto-loaded by Claude)
```

## File Descriptions

### 1. `CLAUDE.md` (Root)
**Purpose**: Primary context file for Claude Code
**Contents**:
- Project overview and goals
- Development environment setup
- Project structure
- Chapter progress status
- Key conventions and patterns
- Common workflows
- References

**When to update**:
- Project structure changes
- New major dependencies
- Overall workflow changes
- Chapter completion milestones

### 2. `MEMORY.md` (Persistent)
**Purpose**: Auto-loaded learnings and patterns
**Location**: `~/.claude/projects/-home-satishthakur-codebase-data-science/memory/MEMORY.md`
**Contents**:
- Core principles (standardization, DAG workflow)
- Common mistakes and solutions
- Statistical intuitions
- Python/PyMC patterns
- Visualization best practices
- Git commit patterns

**When to update**:
- Discover new patterns or anti-patterns
- Learn important statistical insights
- Find better ways to do common tasks
- After completing major sections

**Keep under 200 lines** for efficient loading!

### 3. `TODO.md`
**Purpose**: Track progress and plan next steps
**Contents**:
- Current focus area
- Immediate tasks (checkboxes)
- Chapter status (completed/in-progress/not started)
- Technical debt tracking
- Documentation tasks
- Long-term goals

**When to update**:
- After completing tasks (check them off!)
- When starting new work (add tasks)
- After each chapter
- Weekly reviews

### 4. `notebooks/chapter{N}/CLAUDE.md`
**Purpose**: Chapter-specific context and reference
**Contents**:
- Key concepts covered in the chapter
- Datasets used
- Notebooks in this chapter
- Common patterns for this chapter
- Homework status
- Next steps

**When to update**:
- After creating new notebooks
- After completing major concepts
- After finishing homework
- Before moving to next chapter

### 5. `KNOWLEDGE_GRAPH_SETUP.md` (This File)
**Purpose**: Documentation about the documentation system
**Contents**: How the knowledge graph works and how to maintain it

## Workflows

### Starting a New Session

1. **Claude Code reads automatically**:
   - Root `CLAUDE.md` (via system prompt)
   - `MEMORY.md` (auto-loaded from persistent memory)

2. **You/Claude checks**:
   - `TODO.md` - What to work on next?
   - Relevant `notebooks/chapter{N}/CLAUDE.md` - Chapter context

3. **Start working** with full context already loaded!

### Ending a Session / Before Committing

1. **Update TODO.md**:
   ```bash
   # Check off completed tasks
   - [x] Complete 6H3
   - [ ] Complete 6H4  # If not done
   ```

2. **Update chapter CLAUDE.md** (if new concepts covered):
   ```bash
   # Add to notebooks/chapter6/CLAUDE.md
   - New notebook: `homework_6H3_6H7.ipynb`
   - Status: Problems 6H3-6H5 complete
   ```

3. **Update MEMORY.md** (if learned something important):
   ```bash
   # Add to MEMORY.md
   ## New Pattern: Foxes DAG
   - F → A → W ← G ← F
   - Condition on F only for A → W effect
   ```

4. **Update root CLAUDE.md** (if major milestone):
   ```bash
   # Update chapter status from 🔄 to ✅
   - ✅ **Chapter 6**: Causal inference (COMPLETE)
   ```

5. **Git commit**:
   ```bash
   git add .
   git commit -m "complete chapter 6 homework 6H3-6H5"
   # Hook will remind you about docs if needed!
   ```

### Adding a New Chapter

When starting Chapter 7:

1. **Create directory**:
   ```bash
   mkdir notebooks/chapter7
   ```

2. **Create chapter CLAUDE.md**:
   ```bash
   # Copy template from chapter6/CLAUDE.md
   cp notebooks/chapter6/CLAUDE.md notebooks/chapter7/CLAUDE.md
   # Edit for new chapter content
   ```

3. **Update TODO.md**:
   ```markdown
   ## Chapter 7 Status
   - [ ] Information theory basics
   - [ ] WAIC and LOO
   ...
   ```

4. **Update root CLAUDE.md**:
   ```markdown
   - 🔄 **Chapter 7**: Overfitting & Regularization
     - Currently working on: Information theory
   ```

## Git Hooks

### Installation

```bash
# Install the pre-commit hook
bash .git-hooks/install.sh
```

### What It Does

Before each commit:
1. Detects if notebooks were modified
2. Reminds you to update documentation
3. Warns about large files (>1MB)
4. Suggests good commit message patterns

### Bypass When Needed

```bash
# Skip hooks for quick commits
git commit --no-verify -m "minor fix"
```

## Benefits

### For You
- ✅ Track progress systematically
- ✅ Never forget where you left off
- ✅ Build knowledge incrementally
- ✅ Easy to review what you've learned

### For Claude Code
- ✅ Minimal token usage (efficient context loading)
- ✅ Full project context from day one
- ✅ Knows conventions and patterns
- ✅ Understands what's been done vs what's next
- ✅ Can jump right into work without re-explaining

## Maintenance

### Weekly Review
Every week or after major milestones:

1. Review and update `TODO.md`
2. Check `MEMORY.md` is still concise (<200 lines)
3. Update chapter `CLAUDE.md` files
4. Clean up old/experimental notebooks

### Monthly Archive
Every month:

1. Archive completed notebooks if needed
2. Review and consolidate `MEMORY.md`
3. Update long-term goals in `TODO.md`
4. Backup important work

## Templates

### Chapter CLAUDE.md Template

```markdown
# Chapter N: [Chapter Title]

## Key Concepts
- Concept 1
- Concept 2

## Notebooks
- `notebook1.ipynb` - Description
- `notebook2.ipynb` - Description

## Datasets Used
- Dataset name - Description

## Homework Status
- [ ] Problem 1
- [ ] Problem 2

## Key Principles
...

## Next Steps
...
```

### Commit Message Templates

Good examples:
- `"add [concept] notebook with [specific example]"`
- `"complete chapter N homework problems N-M"`
- `"fix [specific issue] in [file]"`
- `"update quap.py to handle [edge case]"`
- `"chapter N: complete [section name]"`

Bad examples:
- `"update"` (too vague)
- `"fix bug"` (which bug?)
- `"work on chapter 6"` (what specifically?)

## Quick Reference

| Need to... | Update... |
|------------|-----------|
| Start session | Read `CLAUDE.md` + `TODO.md` |
| Learn new pattern | Add to `MEMORY.md` |
| Complete task | Check off in `TODO.md` |
| Finish notebook | Update chapter `CLAUDE.md` |
| Complete chapter | Update root `CLAUDE.md` status |
| Commit | Let git hook remind you! |

## Next Steps

1. ✅ Knowledge graph setup complete (this file)
2. ⏳ Install git hooks: `bash .git-hooks/install.sh`
3. ⏳ Test the system with next commit
4. ⏳ Adjust as needed based on usage

---

*Created: 2026-02-21*
*System ready to use!*
