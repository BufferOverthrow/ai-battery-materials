# AI for Next-Gen Battery Materials
**GNN-Based Discovery, Prediction, and Generative Design of Lithium/Sodium Battery Compounds**

AI Battery Materials is a modular machine learning pipeline designed to explore and predict material properties relevant to next-generation battery research. The current focus is on **band gap prediction** using graph neural networks (GNNs) and other machine learning models.

---
## Scientific Background
### Why Batteries Matter
Batteries are **core to the energy transition** — from EVs and grid storage to mobile devices. Key challenges in lithium-ion and sodium-ion batteries include:
- **Energy density** (how much energy per weight/volume)
- **Cycle life** (how long it lasts before degrading)
- **Safety** (especially avoiding thermal runaway)
- **Cost** (raw materials, scalability, manufacturability)

### Basic Components of a Battery
Every battery has 3 main components:
- **Cathode** (positive electrode): often made of lithium metal oxides like NMC or LFP
- **Anode** (negative electrode): graphite or alternatives like silicon or lithium metal
- **Electrolyte**: transports lithium/sodium ions between cathode and anode

Many cutting-edge efforts focus on **solid-state batteries**, where the electrolyte is a solid rather than a liquid. They promise better safety and energy density — but finding **stable, conductive, and manufacturable** solid-state electrolytes is *hard*.

---
## The Problem
The materials space is **huge** — there are:
- **~10¹⁰⁰ possible molecules** (*way more* than atoms in the universe)
- Countless crystal structures with varying compositions and stoichiometries
  
**Discovering new battery materials** using traditional lab experiments or even DFT (Density Functional Theory) simulations is:
- **Expensive**
- **Slow**
- **Inefficient**

This is known as the **materials discovery bottleneck**.

---
## How AI Can Help

AI — especially **Graph Neural Networks (GNNs)** and **generative models** — can dramatically accelerate the materials discovery pipeline.

a. **GNNs for Property Prediction**
- Molecules and crystals can be modeled as **graphs** (atoms as nodes, bonds as edges).
- GNNs are naturally suited to predict properties from these graph representations.
- Use cases:
   - Predicting **band gap**, **conductivity**, **electrochemical stability**, **formation energy**, etc.
   - Replace or complement DFT simulations.

b. **Generative Models for Material Design**
- **Variational autoencoders (VAEs)**, **diffusion models**, or **graph-based generative models** can learn to generate new candidate materials.
- They can be guided by desired properties (e.g., “generate a crystal with band gap ≈ 4eV and high ionic conductivity”).

c. **Active Learning and Closed-Loop Design**
- AI models can suggest candidates.
- Top ones are simulated (or tested in labs).
- Results are fed back to retrain/improve the model.
- This forms a **closed-loop autonomous discovery system**.

---
## Project Goal

To build a **modular pipeline** that can:
1. **Take structure/property data** from public datasets
1. Train a **GNN** model to predict key properties (band gap, stability, conductivity, etc.)
1. Integrate a **generative model** to propose new materials
1. Optionally simulate top candidates using ASE + pretrained DFT surrogates
1. Package results in a **dashboard or frontend** to explore the generated compounds

---
## Why This Project is Exciting

- Battery innovation is **strategic** to many companies and the clean energy sector.
- AI-first materials discovery reduces R&D timelines from **years to months**.
- We are proposing a **scalable platform**, not just a model — with applications beyond just lithium (e.g., sodium, magnesium, solid-state).
- It plays at the intersection of **chemistry**, **AI**, and **energy** — highly investable deeptech terrain.

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
