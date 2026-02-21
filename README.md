🚀 Agentic Talent Fraud Detection
## 📌 Project Overview
This repository contains an autonomous Active Learning Agent built to identify fraudulent talent profiles. In high-volume recruitment platforms, manually labeling fraud is expensive and time-consuming. Our agent maximizes a strict 100-query budget to achieve a high F1-score by intelligently selecting the most informative data points for human (Oracle) verification.

## 🧠 Methodology: Query-by-Committee (QBC)
Instead of relying on a single model's internal uncertainty, our agent uses an Ensemble Committee strategy. This approach reduces bias and ensures that queries are spent on the most difficult-to-classify examples.

### The Committee
We utilize two different tree-based ensembles:

Random Forest (RF): A bagging ensemble that selects random features and samples with replacement.

Extra Trees (ET): Similar to RF but chooses split points randomly, providing better smoothing and variance reduction.

### The Learning Loop
Initialization: Before the QBC loop starts, an Isolation Forest identifies initial anomalies to ensure the committee has a baseline for fraudulent behavior.

Disagreement Sampling: The agent identifies candidates where the RF and ET models "disagree" most (highest probability delta). These points represent the complex decision boundary.

Iterative Refinement: Queries are conducted in 5 batches. After each batch, the committee is retrained, allowing the agent to "zoom in" on sophisticated fraud rings.

## 🛠️ Feature Engineering (Pillars of Detection)
We engineered high-signal features in helpers.py to expose bot behavior and credential inflation:

Total Risk Score: Aggregates independent risk signals from IP, email, and institution.

Bot Index: Detects automated behavior by crossing application frequency with copy-paste ratios.

Academic Red Flag: Detects credential inflation by comparing GPA anomalies against professional experience.

Velocity Risk: Combines login attempts with failure rates to flag potential account takeovers.

## 📊 Performance Metrics
The agent was evaluated using a local framework simulating the competition environment.

## 📂 Project Structure
## 🚀 Getting Started
### 1. Prerequisites
Python 3.9+

Dependencies: numpy, pandas, scikit-learn, imbalanced-learn

### 2. Installation
### 3. Running the Agent
To test the agent with the competition framework:

## ⚖️ License
This project is licensed under the MIT License - see the  file for details.
