# TODO - Statistical Rethinking Progress Tracker

## Current Focus

**Chapters 7 & 8 Preparation** - Setting up infrastructure and planning systematic coverage

## Immediate Tasks (Priority Order)

### Chapter 6 Wrap-up
- [ ] Complete 6H3-6H7 homework (optional - can defer)
- [ ] Brief review of Chapter 6 concepts

### Chapter 7 - In Progress 🔄
- [x] Create `notebooks/chapter7/` directory
- [x] Create `notebooks/chapter7/CLAUDE.md`
- [x] Read Chapter 7 Section 7.1 (overfitting basics)
- [x] Create `reading_notes.md` for collaborative learning
- [x] Build `polynomial_overfitting.ipynb` (Section 7.1 example)
- [ ] **Enhance quap.py with WAIC support** (next priority)
  - [ ] Add `log_likelihood()` method
  - [ ] Add `waic()` method
  - [ ] Add `compare_models()` function
  - [ ] Write tests for WAIC implementation

## Chapter 6 Status

### Completed ✅
- [x] Multicollinearity concept (legs example, milk example)
- [x] Fixed multicollinearity visualization (6 plots showing uncertainty)
- [x] Collider bias notebook
- [x] Post-treatment bias notebook
- [x] Setup homework notebook for 6H3-6H7

### In Progress 🔄
- [ ] Finish homework problems 6H3-6H7
- [ ] Review and test all solutions

### Not Started ⏳
- [ ] Chapter 6 summary/review notebook
- [ ] Additional practice problems
- [ ] DAG practice exercises

## Chapter 7: Overfitting & Model Comparison

**Goal**: 2-3 sessions, comprehensive coverage
**See**: `CHAPTERS_7_8_PLAN.md` for detailed structure

### Session 1: Information Theory & WAIC
- [ ] Notebook: `information_theory_basics.ipynb`
- [ ] Notebook: `waic_model_comparison.ipynb`
- [ ] Understand entropy, KL divergence, deviance
- [ ] Master WAIC calculation and interpretation

### Session 2: Cross-Validation & Regularization
- [ ] Notebook: `cross_validation.ipynb`
- [ ] Notebook: `regularization.ipynb`
- [ ] Implement LOO-CV (or integrate ArviZ)
- [ ] Understand regularizing priors

### Session 3: Practice & Review (Optional)
- [ ] Notebook: `chapter7_homework.ipynb`
- [ ] Chapter review and consolidation

## Chapter 8: Interactions

**Goal**: 2-3 sessions, comprehensive coverage
**See**: `CHAPTERS_7_8_PLAN.md` for detailed structure

### Session 1: Categorical Interactions
- [ ] Notebook: `interactions_intro.ipynb`
- [ ] Notebook: `categorical_categorical_interactions.ipynb`
- [ ] Understand interaction concepts
- [ ] Master categorical × categorical

### Session 2: Continuous Interactions
- [ ] Notebook: `continuous_continuous_interactions.ipynb`
- [ ] Notebook: `categorical_continuous_interactions.ipynb`
- [ ] Master continuous × continuous
- [ ] Master categorical × continuous

### Session 3: Practice & Review (Optional)
- [ ] Notebook: `chapter8_homework.ipynb`
- [ ] Chapter review and consolidation

## Chapter 9+
- Not yet planned

## Technical Debt

- [ ] Add automated tests for `quap.py`
- [ ] Create helper functions for common plotting patterns
- [ ] Organize notebooks/ with README files per chapter
- [ ] Archive old/experimental notebooks properly

## Documentation

- [x] Create comprehensive root CLAUDE.md
- [x] Create MEMORY.md with key learnings
- [x] Create TODO.md (this file)
- [ ] Create chapter-specific CLAUDE.md files:
  - [ ] notebooks/chapter2/CLAUDE.md
  - [ ] notebooks/chapter3/CLAUDE.md
  - [ ] notebooks/chapter4/CLAUDE.md
  - [ ] notebooks/chapter5/CLAUDE.md
  - [x] notebooks/chapter6/CLAUDE.md (to be created)
- [ ] Setup git hooks for auto-updating docs

## Long-term Goals

- [ ] Complete all Statistical Rethinking chapters 2-16
- [ ] Build reusable library of Bayesian modeling utilities
- [ ] Create tutorial notebooks for key concepts
- [ ] Contribute examples back to PyMC community

## Notes

- Focus on understanding concepts deeply, not rushing through
- Always work through homework problems to solidify understanding
- Update this file after completing each major section
- Keep MEMORY.md updated with new patterns and learnings

---

*Last updated: 2026-02-21*
*Next milestone: Complete Chapter 6 homework*
