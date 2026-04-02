# Propensity to Trust in LLMs

This repository contains the code and dataset for the paper *"Propensity to Trust in Large Language Models"* accepted for publication on PLOS One.

This project implements a multi-agent simulation framework for studying propensity to trust (PTT) in large language models (LLMs), following the methodology introduced in “Propensity to trust in large language models.” Trust is a core mechanism enabling coordination and delegation in collaborative settings, yet it remains unclear whether LLMs exhibit stable, context-independent trust behaviors. This work operationalizes PTT as a baseline tendency to delegate tasks under uncertainty, and evaluates it through language-mediated simulations in which agents interact, form beliefs about others’ trustworthiness, and update decisions based on observed outcomes. Unlike questionnaire-based approaches—which tend to reflect alignment-driven, socially desirable responses—this framework captures behavioral trust dynamics, revealing how models balance intrinsic delegation tendencies with sensitivity to evidence about collaborators’ capability, reliability, and willingness.

## 📁 Project Structure

```
~/
│
├── src/
│   ├── main_exec.py        # Main entry point
│   ├── simulation.py       # Core simulation loop
│   ├── agent.py            # Agent logic and behavior
│   ├── models.py           # Model definitions and interfaces
│   ├── lm.py               # LLM interaction layer
│   ├── trustq.py           # Trust computation logic
│   ├── logic.py            # Decision logic
│   ├── complete.py         # Completion handling
│   ├── plot.py             # Visualization utilities
│   ├── do_plots.py         # Plot execution scripts
│   ├── compare.py          # Result comparison tools
│   ├── scan_res.py         # Result scanning/aggregation
│   ├── load_cnfg.py        # Configuration management
│   ├── cfg_00.py           # Example configuration
│   ├── heat_sim.py         # Heatmap simulation
│   ├── heat_qst.py         # Heatmap analysis
│   └── infstat.py          # Statistical analysis
│
└── data/
    ├── agents.json         # Agent definitions
    ├── dialogs.json        # Dialogue templates
    ├── frazier.json        # Dataset / benchmark
    ├── scenario_fire.json
    ├── scenario_farm.json
    └── scenario_school.json
```




## ⚙️ Requirements
This project uses `Python 3.12.3`. You can install dependencies via:

```
$ pip install -r requirements.txt
```

<!--
## 📦 Dataset

The dataset of 200 news items is available at the following link:

🔗 [Download 200-News Dataset (Google Drive)](https://drive.google.com/drive/folders/1U3HPyt4NktwLExbcwWQZNLCH4uMwiyW_)

After downloading, place the `imgs/` folder at the same level as the `src/` and `data/` directories.

The corresponding text data (`news_200.json`) is already included in the `data/` directory.

**Disclaimer:** This dataset contains material (such as text and images) that may be protected by copyright and owned by third parties. We do not claim any rights over such content. All copyrights remain with their respective owners. This dataset is shared solely for non-commercial research and educational purposes.

-->

## 🚀 Running the Code

To view available command-line arguments, use the `-h` flag:

```
$ python3 main_exec.py -h
```

Use the `-v` option to visualize simulation progress across news items.

More detailed configuration parameters can be passed through a configuration file, such as `cnfg_00.py`, using the `-c` option.

```
$ python3 main_exec.py -c cnfg_00 -v
```

## 📄 License

Released under the MIT License.


