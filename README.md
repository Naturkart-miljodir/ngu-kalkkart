# ngu-kalkkart | Mapping of lime-rich areas using remote sensing

<!-- PLEASE fill out this README template in english.
For an example see: https://github.com/miljodirektoratet/uv-demo
Replace title with format: "<repo-name> | <Project Full Name in English>" -->

<!-- LICENSE BADGES: Update or add badges below as needed.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Please fill out this README.md template folowing the instructions (view raw file to see comments). Remove all instructions and comments when done.

**Norwegian Title:** Kalkkart  
**Project Owner:** Norwegian Environment Agency (NEA)  
**Contractor:** The Geological Survey of Norway (NGU)  
**Delivery date:** 2026.12.01  
**Repository Purpose:** _Short description of the purpose of this repository_

---

**Table of Contents**

<!-- AUTO-GENERATED TOC: Update this table of contents by the vscode-extension "yzhang.markdown-all-in-one" -->

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Development Workflow](#development-workflow)
- [Documentation](#documentation)
- [License](#license)

## Project Overview

**English:**

**Norwegian:**

---

### Key Features

- ...
- ...

### Resulting Products

_List the main resulting datasetts, applications, or other products resulting from this project._

| Product | Description | URL |
| ------- | ----------- | --- |
|         |             |     |

### Repository Structure

_Provide an overview of the repository structure here._

```text
nk-template/
├── src/
├── LICENSE
└── README.md
```

## Prerequisites

_Document all prerequisites needed to run the analysis and reproduce the results. Complete the relevant sections below and remove any sections that do not apply._

- **Hardware:**
  - **OS:** [name, version/build]
  - **Memory (RAM):** [total GB, speed]
  - **CPU:** [model, total cores/threads]
  - **GPU:** [model, VRAM, driver/CUDA]
  - **Storage:** [type, size]
  - **Performance:** _Preferably specify expected runtime or benchmark tests for the analyses._
- **Runtime environment:**
  - **Container:** _Dockerfile, docker compose, and other relevant config files._
  - **Local PC:** _OS and version/build, key system dependencies (e.g., GDAL), and any other dependencies required by Python/R packages._
  - **Databricks:** _Databricks Runtime version, any cluster init scripts, or policy requirements._
  - **Other runtime environments:** _Describe relevant configurations and system requirements._
- **Software:**
  - [Software (version ≥ x.x)]
  - _Document all required software dependencies and versions (QGIS, ArcGIS Pro, E-cognition, FME, etc.)._
- **AI tools:**
  - [AI tool (model + version ≥ x.x)]
  - _Document all AI tools and versions used (ChatGPT, Claude, etc.)._
  - _Include all relevant instruction files (e.g., `AGENTS.md` or `CLAUDE.md`) in the repository._
  - _Briefly describe how the AI tools were used (e.g., documentation, analysis, code generation)._
- **Programming languages:**
  - [Programming language (version ≥ x.x)]
  - _Document all programming languages and versions used (Python, R, SQL, etc.)._
- **Analysis environment definition:**
  - All required packages, dependencies, and versions must be specified in an environment file stored in the repository.
  - The environment file must be sufficient to recreate the analysis environment without manual installation or manual package/version specification.
  - `renv.lock` (_recommended for R_), `pyproject.toml` (_recommended for Python_), `environment.yml`, `pixi.toml`, or `requirements.txt`.

  **R environment:**
  - We recommend using `renv` to manage packages and dependencies.
  - Any system dependencies required to build or run packages (e.g., GDAL, PROJ, and GEOS for geospatial packages) must be documented.
  - If installing system dependencies requires dedicated commands, we recommend including a script (e.g., `.sh`) that documents the installation steps.
  - R version must be >4.4.

  **Python environment:**
  - We recommend using `uv` as the package manager and defining the environment in `pyproject.toml`.
  - If conda support is required, we recommend using `pixi`.
  - See the [uv-demo](https://github.com/miljodirektoratet/uv-demo) repository for recommended practices for Python development and reproducible environments.
  - Python version must be >3.11.

  **ArcPy environment:**
  - For analyses using ArcPy, ArcGIS Pro's conda environment is recommended.
  - The environment should be exported to an `environment.yml` file stored in the repository.
  - See [gis-utils-public](https://github.com/miljodirektoratet/gis-utils-public) for recommended practices for ArcPy development and reproducible environments.

## Installation

_Provide installation instructions here._

## Development Workflow

_Provide instructions for the typical development workflow here (e.g. in which order to run scripts, how to test changes and how to make contributions)._

## Documentation

_Provide details about documentation here or refer to the /docs directory._

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
