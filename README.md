# AI Battery Materials

AI Battery Materials is a modular machine learning pipeline designed to explore and predict material properties relevant to next-generation battery research. The current focus is on **band gap prediction** using graph neural networks (GNNs) and other machine learning models.

---

## Features
1. Data ingestion and preprocessing of crystal/material structures.  
2. Band gap prediction using ML/DL models (starting with GNNs).  
3. Modular design for extending to other properties in the future (ionic conductivity, stability, etc.).  
4. Training and evaluation scripts.  
5. Clear separation of raw data, processed data, and trained models.  

---

## Repository Structure

```
ai-battery-materials/
│
├── data/                   # Datasets (raw and processed)
│   ├── raw/                # Original datasets (e.g., Materials Project, OQMD)
│   └── processed/          # Preprocessed datasets for training
│
├── models/                 # Saved models and checkpoints
│
├── notebooks/              # Jupyter notebooks for experimentation
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── dataloader.py       # Scripts for dataset loading, preparation, and basic preprocessing
│   ├── model.py            # Model architectures (e.g., GNN, baseline ML)
│   ├── train.py            # Training loop
│   └── evaluate.py         # Evaluation metrics and testing
│
├── tests/                  # Unit tests
├── main.py                 # Entry point script to run the full pipeline (data loading, training, evaluation, or inference)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BufferOverthrow/ai-battery-materials.git
   cd ai-battery-materials
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Linux/Mac
   .venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Preprocess and train the model:
   ```bash
   python main.py
   ```

2. Evaluate the model:
   ```bash
   python src/evaluate.py 
   ```

---

## Roadmap

- [x] Initialize project structure  
- [ ] Add preprocessing pipeline for Materials Project data  
- [ ] Implement baseline ML model (Random Forest, XGBoost)  
- [ ] Implement GNN-based band gap predictor  
- [ ] Build evaluation and visualization tools  
- [ ] Extend to additional material properties  

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for discussion.  

---

## License

This project is currently **unlicensed**.  
You may fork and experiment, but redistribution is discouraged until a license is added.  
