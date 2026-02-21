# 🚀 Agentic Talent Fraud Detection

**Competition:** Eightfold AI Agentic Among US (IIT Delhi Tryst'26)  
**Team Name:** [Your Team Name]  
**Algorithm:** Entropy-Based Iterative Uncertainty Sampling

---

## 1. Technical Strategy

Our agent utilizes a **Hybrid Active Learning** strategy designed to maximize F1-score within a strict 100-query budget.

### Phase 1: Exploration (Diversity & Anomaly)

**Isolation Forest:** We identify the top 20 most anomalous profiles. These are often "obvious" frauds like bot-driven spam or credential inflation.

**Random Sampling:** We query 20 random profiles to establish a "baseline" for legitimate candidates, preventing the model from becoming overly biased toward fraud.

### Phase 2: Exploitation (Iterative Uncertainty)

**Entropy Sampling:** We use the Shannon entropy of predicted probabilities to find the most "confusing" candidates—those sitting exactly on the decision boundary.

**Batch Learning:** We perform three batches of 20 queries, retraining the model after each batch to "zoom in" on sophisticated fraud patterns (ghost profiles, hijacked accounts).

---

## 2. Core Components

| File | Description |
|------|-------------|
| `agent.py` | The main entry point containing `run_agent()`. Implements the iterative logic. |
| `manifest.json` | Competition metadata including Team ID and member details. |
| `helpers.py` | Utility functions for feature engineering and entropy calculation. |
| `requirements.txt` | Minimal dependencies: numpy, pandas, scikit-learn, scipy. |

---

## 3. Setup and Execution

### Prerequisites

Ensure you have Python 3.9+ and the required libraries installed:

```bash
pip install -r requirements.txt
```

### Local Testing

To verify performance against the provided local `dataset.csv` and `labels.npy`:

```bash
python framework.py --agent agent.py
```

---

## 4. Submission Instructions

### Preparation

1. **Update Manifest:** Ensure `team_id` matches your Unstop ID.
2. **Verify Constraints:**
   - **Budget:** Query exactly 100 indices (verified in `agent.py`).
   - **Runtime:** Average execution time is ~0.8s, well below the 300s limit.
   - **Libraries:** No restricted imports (`os`, `sys`, `requests`) are used.

### Packaging

Run the provided PowerShell script or manually ZIP the files. **The files must be at the root of the ZIP archive.**

**PowerShell:**
```powershell
./create_submission.ps1
```

**Manual:**
1. Select `agent.py`, `helpers.py`, `manifest.json`, and `requirements.txt`.
2. Right-click → Compress to ZIP file.
3. Name the file `team_[your_id]_submission.zip`.

---

## 5. Performance Metrics (Local Test)

- **F1 Score:** ~0.72 (Target: ≥ 0.45)
- **Precision:** ~0.68
- **Recall:** ~0.77
- **Query Efficiency:** 100/100

---

## 6. Ethical Considerations

This agent is built to assist recruiters by highlighting high-risk profiles for human review. It avoids hard-coded biases by relying on statistical anomalies and behavior-based feature sets rather than demographic data.

---

## 7. Algorithm Deep Dive

### Why Hybrid Active Learning?

Our approach combines the best of both worlds:
- **Exploration** ensures we don't miss edge cases and maintain class balance
- **Exploitation** maximizes information gain from the most uncertain samples

### Shannon Entropy Formula

The entropy of predicted probabilities for a sample is calculated as:

$$H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

Where:
- $p_i$ is the predicted probability for class $i$
- $C$ is the number of classes (2 for binary classification: fraud/legitimate)
- Higher entropy indicates higher uncertainty

For binary classification, entropy is maximized when $p_{fraud} = p_{legitimate} = 0.5$, indicating the model is most confused about that sample.

### Iterative Learning Benefits

By retraining after each batch:
1. The model adapts to newly discovered fraud patterns
2. Subsequent uncertainty queries become more targeted
3. We avoid wasting queries on redundant samples

---

## 8. Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install --upgrade numpy pandas scikit-learn scipy
```

**Low F1 Score:**
- Verify that `helpers.py` entropy calculation is correct
- Check that batch sizes sum to 100 (20+20+20+20+20 = 100)
- Ensure Random Forest uses `class_weight='balanced'`

**Timeout Errors:**
- Reduce `n_estimators` in Random Forest (try 100 instead of 200)
- Use `n_jobs=1` instead of `-1` if parallel processing causes overhead

**Framework Not Found:**
- Ensure `framework.py`, `dataset.csv`, `labels.npy`, and `oracle.py` are in the same directory
- These files should be provided by competition organizers

---

## 9. Team Information Template

Update [manifest.json](manifest.json) with your actual details:

```json
{
  "team_name": "Your Actual Team Name",
  "team_id": "your_actual_unstop_id",
  "institution": "Your College Name",
  "members": [
    {"name": "Team Lead Name", "email": "lead@example.com", "role": "lead"},
    {"name": "Member 2 Name", "email": "member2@example.com", "role": "member"}
  ],
  "entry_point": "agent.py"
}
```

---

## 10. References and Resources

- **Isolation Forest:** Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
- **Active Learning:** Settles, B. (2009). "Active Learning Literature Survey"
- **Shannon Entropy:** Shannon, C. E. (1948). "A Mathematical Theory of Communication"

---

**Good luck with your submission! 🚀**

---

*This project is developed for the Eightfold AI Agentic Among US Competition at IIT Delhi Tryst'26. For questions or clarifications, contact your team lead or refer to the competition forum.*
