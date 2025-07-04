
 n a m e n o t f o u n d . a i            

 T H E      
 F U T U R E      
 O F      
 A I      
 I S     
 O N - D E V I C E                     


> **A-Agent** — An AI-powered workspace system for creation

---

## Project Overview

**namenotfound.ai** This system is a local-first AI system designed to build elements. 
For instance instead of saying "Build a pitch deck" This system would build an app that builds pitch decks. 
You act as a owner, build chain flows and roles. The underlying model engines can be swapped, you can use your own or an api. 
This system can run locally on any hardware however the hardware does determine how long a chain can be and the needed adjustments to the models.
Chains and orgs can be saved and shared so you can share anything you wish, additionally everything is as private and on device as you want it to be!




## Tutorial:

https://www.loom.com/share/f19ac59a21ad4fd4b422552e958b3385?sid=1e26e48d-2643-4003-9288-1e65eb02de0f

Backup Download Link for Models:

https://drive.google.com/file/d/1-e6Hd-qYOZ0v7caDwxxkRFoOzb3X101G/view?usp=sharing


## Installation:

Use Conda
```bash

conda create -n a_agent_2_python_3_11 python=3.11
conda activate a_agent_2_python_3_11

conda install -c conda-forge tensorflow keras numpy pandas matplotlib scikit-learn

# Create a new requirements file without the packages you installed via conda
grep -v -E "(tensorflow|keras|numpy|pandas|matplotlib|scikit-learn)" requirements.docker.txt > requirements_clean.txt

# Install the cleaned requirements
pip install -r requirements_clean.txt
```




