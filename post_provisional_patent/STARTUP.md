# STARTUP GUIDE - Post-Provisional Patent Work

**Quick start guide for running Test #12 and future post-provisional tests**

---

## 📂 Data Location

**Test #12 requires the Lending Club dataset:**

**Location:** `~/Desktop/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`

**Note:** This is a 1.6GB file (2.2M loans) and is NOT included in this repo. Download from:
- Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Place in: `~/Desktop/archive/`

---

## 🚀 Running Test #12
```bash
# 1. Navigate to validation scripts
cd ~/Desktop/task-identity/post_provisional_patent/validation_scripts

# 2. Activate environment (if needed)
# source ../../task-identity-env/bin/activate

# 3. Run the test
python financial_lending_test.py
```

**Expected runtime:** 5-10 minutes (training on 2.2M loans)

**Expected output:**
- Task-Identity Overall: ~0.921
- Task-Identity Default Class: ~0.000
- Detection Gap: ~92.1 points

---

## 📊 Results Location

Results saved to: `post_provisional_patent/results/12_financial_lending/`

**Format:** JSON files with timestamp (e.g., `lending_test_20251104_132749.json`)

---

## 🔍 What Test #12 Validates

**Scenario:** Loan default prediction model becomes overly conservative

**Method:** Per-class Task-Identity (Test #4 approach)

**Key Finding:** 
- Standard method missed catastrophic drift (0.921)
- Per-class method detected it (0.000)
- 92.1 percentage point improvement

**Why This Matters:** Demonstrates Task-Identity works on real financial data with extreme class imbalance when using per-class analysis.

---

## 📁 Folder Structure
```
post_provisional_patent/
├── STARTUP.md                          (this file)
├── README.md                           (detailed documentation)
├── validation_scripts/
│   └── financial_lending_test.py       (Test #12 code)
├── results/
│   └── 12_financial_lending/          (test results)
└── datasets/                           (empty - data stored elsewhere)
```

---

## ⚠️ Important Notes

1. **Data Not Included:** 1.6GB Lending Club CSV must be downloaded separately
2. **Real Data Only:** This test uses 100% real financial data (no synthetic)
3. **Reproducible:** Fixed random_state=42 ensures consistent results
4. **Runtime:** ~5-10 minutes on full dataset

---

## 🆘 Troubleshooting

**Error: File not found**
- Ensure Lending Club data is in `~/Desktop/archive/`
- Check path: `../../archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`

**Error: Module not found**
- Install: `pip install pandas numpy scikit-learn`
- Or activate: `source ../../task-identity-env/bin/activate`

**Test runs but gives different results**
- This is expected if data location/version differs
- Core result (per-class detects drift) should remain consistent

---

**Last Updated:** November 4, 2025  
**Test Status:** ✅ Validated and passing
