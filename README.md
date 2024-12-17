# Urban Computing Final

## Overview
This repository contains transportation datasets and a Jupyter Notebook to perform analysis on **bus, subway, and taxi data**. The notebook can be run directly to execute all tasks, including data preprocessing, visualization, and modeling.

---

## Folder Structure
- **`datasets/`**: Contains all data files related to bus, subway, and taxi transportation.
    - Source Links:
        - NYC Taxi Data: [New York City Taxi and Limousine Commission (TLC) Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
        - GTFS Data: [MTA Subways and Buses GTFS Data](https://new.mta.info/developers)
        - Taxi Zones: [NYC Open Data - NYC Taxi Zones](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc)

- **`UrbanComputing_FinalProject.ipynb`**: Jupyter Notebook that performs all the analysis. Execute this notebook to run the full workflow.

- **`.gitattributes`**: Configurations for Git LFS to handle large datasets.

---
## How to Run
1.	Clone this repository:
```bash
git clone git@github.com:becoollll/Urban-Computing-Final.git
```

2.	Install the dependencies:
```bash
pip install -r requirements.txt
```

3.	Open the notebook:
```bash
jupyter notebook UrbanComputing_FinalProject.ipynb
```
