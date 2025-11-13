# Latin Square Solver (CSP)

This Python program solves **partially filled Latin squares** using **constraint satisfaction techniques**:  
- **MRV (Minimum Remaining Values)**
- **Forward Checking**
- **LCV (Least Constraining Value)**

---

## Problem

A Latin square of size `n` is an `n × n` grid filled with numbers `1..n` such that:  
- Each number appears **exactly once per row**  
- Each number appears **exactly once per column**

The solver completes a partially filled grid while respecting these constraints.

---

## Features

- Handles **user input grids** or generates random partial grids.  
- Checks for **initial conflicts** before solving.  
- Uses CSP techniques:
  - MRV for variable selection  
  - LCV for value ordering  
  - Forward checking to prune domains  
- Supports a **timeout** to stop long searches.  
- Tracks **assignments** and **backtracks** statistics.


```bash
python latin_csp.py
