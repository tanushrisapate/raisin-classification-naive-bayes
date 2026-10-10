# Raisin Variety Classification Using Gaussian Naive Bayes

Machine Learning Lab 7: classify **Kecimen** and **Besni** raisins using seven numerical characteristics extracted from raisin images.

## What this lab covers

- Reading the Raisin dataset and inspecting missing and duplicate values
- Encoding the `Class` target into numeric labels
- Stratified 80:20 train-test split
- Gaussian Naive Bayes (`GaussianNB`) classification
- Accuracy, precision, recall, F1-score, confusion matrix and classification report

## Project structure

```text
raisin-classification-naive-bayes/
├── data/
│   └── Raisin_Dataset.xlsx  (download if needed)
├── notebook/
│   └── Raisin_Gaussian_Naive_Bayes.ipynb
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

**Dataset:** Download `Raisin_Dataset.xlsx` from the [UCI Raisin dataset](https://archive.ics.uci.edu/dataset/850/raisin) and place it in `data/`. If it is not there, the script will try to retrieve the same dataset from UCI using `ucimlrepo` (internet connection required).

For the notebook, open `notebook/Raisin_Gaussian_Naive_Bayes.ipynb` and run all cells in order.

## Important evaluation note

The original classroom notebook selected all numeric columns as input, **including `Class_encoded` (the target itself)**. This causes target leakage and makes its previously saved **82.78% accuracy** unreliable as an unbiased test result. The GitHub script and cleaned notebook **exclude `Class_encoded` from `X`**. Rerun them to generate valid evaluation metrics; these may differ from the old saved values.

## Dataset credit

Çinar, İ., Koklu, M., & Tasdemir, S. (2020). *Raisin* [Dataset], UCI Machine Learning Repository. [DOI: 10.24432/C5660T](https://doi.org/10.24432/C5660T). Dataset is shared under **CC BY 4.0**.

## Author

**Shreyank Nipane**  
B.Tech CSE (Artificial Intelligence & Machine Learning)
