# ChemoBrain Analysis

This repository contains analysis scripts and data for the ChemoBrain project, focusing on the analysis of feeding behavior, running wheel activity, and weight measurements in response to chemotherapy treatment.

## Project Overview

This project analyzes various behavioral and physiological parameters in response to chemotherapy treatment, including:
- Feeding behavior analysis (FED)
- Running wheel activity
- Weight measurements
- Circadian rhythm analysis

## Repository Structure

```
.
├── FED_analysis.ipynb          # Feeding behavior analysis
├── RunningWheel_analysis.ipynb # Running wheel activity analysis
├── Weight_analysis.ipynb       # Weight measurements analysis
├── Plots/                      # Generated plots and visualizations
│   ├── FED/                   # Feeding-related plots
│   ├── RW/                    # Running wheel plots
│   └── Weight/                # Weight-related plots
└── myenv/                     # Python virtual environment
```

## Analysis Components

### 1. Feeding Behavior Analysis
- Analysis of feeding patterns
- Breakpoint analysis
- Pellet consumption tracking
- Poke behavior analysis
- Circadian rhythm analysis of feeding

### 2. Running Wheel Analysis
- Activity pattern analysis
- Circadian rhythm analysis
- Sex-based group comparisons
- Multi-day analysis

### 3. Weight Analysis
- Weight tracking over time
- Group comparisons
- Treatment effect analysis

## Setup and Environment

The project uses a Python virtual environment (`myenv/`). To set up the environment:

1. Create a new virtual environment:
```bash
python -m venv myenv
```

2. Activate the environment:
```bash
source myenv/bin/activate  # On Unix/macOS
```

3. Install required packages (requirements.txt will be added soon)

## Usage

1. Ensure you have the required Python environment set up
2. Open the relevant Jupyter notebooks for your analysis
3. Run the analysis cells in sequence
4. Generated plots will be saved in the Plots directory

## Data Organization

The analysis is organized by behavioral measure:
- FED (Feeding) data
- Running Wheel data
- Weight measurements

Each analysis type has its own notebook and corresponding plots directory.

## Contributing

This is a private repository for research analysis. Please contact the repository owner for access or collaboration.

## License

Private repository - All rights reserved