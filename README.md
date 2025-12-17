# SDET-Intern-Assessment

## Table of content
- [Overview](#Overview)
- [Authorization](#Authorization)
- [Run](#Run)

## Overview
Write a script in Python or JavaScript that: 
- 1. Reads both CSV files
- 2. Compares them based on a primary key
- 3. Detects: missing rows (rows present in one file but not the other), mismatched values in same fields
- 4. Outputs: a simple report (console, CSV, or word, excel) including: total mismatches, total missing rows, details of mismatched fields
<p align="center">
  <img src="https://github.com/Eel-Aig-gYuh/SDET-Intern-Assessment/blob/main/assert/Screenshot%202025-12-17%20103932.png"/>
</p>

## Run
Step 1: clone project
```bash
git clone https://github.com/Eel-Aig-gYuh/SDET-Intern-Assessment.git
```

Step 2: Install requirements
```bash
pip install -r requirements.txt
```

Step 3: Run project
```bash
cd SDET-Intern-Assessment
uvicorn api.data_diff_api:app --reload
```

Step 4: Go http://127.0.0.1:8000/ to access the web

## Authorization
Le Gia Huy
