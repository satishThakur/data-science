# Bayesian Data Analysis — Notes

These notes are a faithful digitisation of handwritten notes, structured exactly in the order they were developed.
They focus on intuition, joint distributions, marginalisation, and a complete worked Bayesian example (Panda problem).

---

## 1. Probability and Uncertainty

Probability is a way to represent **uncertainty in parameters**, arising due to:
- Measurement errors
- Imperfections in the model itself

### Multiple conjectures / hypotheses
- We can have multiple conjectures (hypotheses)
- All conjectures are considered *simultaneously*
- They compete at the same time

> A conjecture that produces the data in more ways is more probable

### Interpretation of probability
Probability here is **not long‑run frequency**.
It follows the **Laplace–Jeffreys–Cox–Jaynes** interpretation:
> Probability as degree of plausibility.

---

## 2. Core Bayesian Concepts

### Parameter
Every discrete value of a parameter is a **conjecture**.

### Likelihood
- Relative number of ways a parameter value can produce the data
- Measures how compatible the data is with a conjecture

$$P(D \mid \theta)$$

### Prior
- Prior possibilities of all parameter values
- Encodes plausibility **before seeing data**

$$P(\theta)$$

### Posterior
- Updated probability of parameter values
- After conditioning on observed data

$$P(\theta \mid D)$$

---

## 3. Discrete Parameter Case

Let:

$$\theta \in \{\theta_1, \theta_2, \dots, \theta_n\}$$

Posterior:

$$P(\theta_i \mid y) = \frac{P(y \mid \theta_i) P(\theta_i)}{P(y)}$$

Where:

$$P(y) = \sum_i P(y \mid \theta_i) P(\theta_i)$$

### Interpretation of denominator
- Average likelihood over the prior
- Also called **expected likelihood**
- Also called **marginalisation over $\theta$**

---

## 4. Likelihood as a Generative Object

Bayesian models are **generative**.

### Two views of likelihood

**P₁: Data → Parameter**
- Given data, how compatible is it with each parameter?
- Used in posterior computation

**P₂: Parameter → Data**
- Given parameter, probability of observing data
- Used in prediction

Prediction = **averaging likelihood over parameter distribution**.

---

## 5. Continuous Parameters

For continuous parameters:

$$P(\theta \mid y) = \frac{P(y \mid \theta) P(\theta)}{\int P(y \mid \theta) P(\theta) \, d\theta}$$

- Integral replaces summation
- Still marginalisation over $\theta$

### Joint distribution
The joint distribution contains **complete information**.

From joint, we obtain:
- Marginal distributions
- Predictive distributions

---

## 6. Predictive Distributions

### Prior Predictive
Before observing data:

$$P(y) = \int P(y \mid \theta) P(\theta) \, d\theta$$

(average likelihood over prior)

### Posterior Predictive
After observing data:

$$P(y_{new} \mid y) = \int P(y_{new} \mid \theta) P(\theta \mid y) \, d\theta$$

(average likelihood over posterior)

---

# Worked Example: Panda Problem

## Problem Statement

- Two panda species: **A** and **B**
- Both equally common in the wild
- Twin birth rates:
  - $P(T \mid S_A) = 0.1$
  - $P(T \mid S_B) = 0.2$
- Genetic test:
  - Identifies $S_A$ with probability 0.8
  - Identifies $S_B$ with probability 0.65

We observe:
1. A panda gives birth to twins
2. A genetic test result

Questions:
- What is the probability the panda is $S_A$ or $S_B$?
- What is the probability the next birth is a twin?
- How do these change with new evidence?

---

## Step 1: Prior

$$P(S_A) = 0.5, \quad P(S_B) = 0.5$$

---

## Step 2: Posterior After Twin Birth

$$P(S_A \mid T) = \frac{0.1 \times 0.5}{0.1 \times 0.5 + 0.2 \times 0.5} = 0.33$$

$$P(S_B \mid T) = 0.67$$

**Intuition**: Twin birth favours species B.

---

## Step 3: Posterior Predictive After Twin Birth

$$P(T_{next}) = 0.1 \times 0.33 + 0.2 \times 0.67 = 0.165$$

Prior predictive was:

$$0.1 \times 0.5 + 0.2 \times 0.5 = 0.15$$

---

## Step 4: Genetic Test Evidence

Test characteristics:

$$P(A^+ \mid S_A) = 0.8$$

$$P(A^+ \mid S_B) = 0.35$$

Updated priors (from previous posterior):

$$P(S_A) = 0.33, \quad P(S_B) = 0.67$$

---

## Step 5: Posterior After Genetic Test

$$P(S_A \mid A^+) = \frac{0.8 \times 0.33}{0.8 \times 0.33 + 0.35 \times 0.67} = 0.533$$

$$P(S_B \mid A^+) = 0.467$$

---

## Step 6: Updated Posterior Predictive

$$P(T_{next}) = 0.533 \times 0.1 + 0.467 \times 0.2 = 0.1467$$

---

## Final Summary

### Parameter evolution

- $S_A$: 0.5 → 0.33 → 0.533
- $S_B$: 0.5 → 0.67 → 0.467

### Prediction evolution

$$P(T_{next}) : 0.15 \rightarrow 0.165 \rightarrow 0.1467$$

### Core takeaway

> Evidence updates parameter plausibility.
> Updated parameters update predictions.

---
