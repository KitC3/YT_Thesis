# The Adaptive Power of Video Thumbnails in the Age of Algorithmic Curation

## Description

This repository accompanies the master thesis **"The Adaptive Power of Video Thumbnails in the Age of Algorithmic Curation"**. It provides a complete workflow for collecting YouTube video metadata, analyzing video thumbnails, and conducting advanced statistical modeling. The pipeline is designed for research projects involving YouTube content, visual analytics, and causal inference.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Python Workflow](#python-workflow)
  - [R Workflow](#r-workflow)
- [Tests](#tests)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contact](#contact)

---

## Installation

1. **Clone the repository:**
  ```
  git clone https://github.com/KitC3/YT_Thesis.git
cd YT_Thesis
  ```

3. **Python Environment:**
- Python 3.8+ recommended.
- Install required Python packages:
  ```
  pip install -r requirements.txt
  ```
- Some scripts require additional models/data (e.g., YOLO weights, DeepFace models). See comments in scripts for setup.

3. **R Environment:**
- R version 4.2+ recommended.
- Install required R packages (see each R script for details):
  ```
  install.packages(c("tidyverse", "MatchIt", "fixest", "cobalt", "nnet", "broom", "ggeffects", "ggplot2", "optmatch"))
  ```

4. **API Credentials:**
- Obtain a YouTube Data API key from the Google API Console.
- Insert your API key in the relevant Python scripts (see comments for `YOUTUBE_API_KEY` or `API_KEY` variables).

---

## Usage

### Python Workflow

The main workflow consists of six core Python scripts (run in order):

1. **01-Initial-Youtube-Channel-Collection.py**  
Collects initial YouTube channel IDs and metadata.
2. **02-Retrieve-All-Channel-Videos.py**  
Retrieves all videos and thumbnails for selected channels.
3. **03-Thumbnail-Analysis.py**  
Extracts visual, color, object, face, and text features from thumbnails.
4. **04-Preprocessing-Feature-Engineering.py**  
Merges and engineers features for modeling.
5. *(Optional)* **Data-Exploration-Plotting.ipynb**  
Jupyter notebook for exploratory data analysis and visualization.
6. *(Optional)* **Thumbnail-Analysis-Overlays.ipynb**  
Visualizes overlays of thumbnail analysis.

### R Workflow

The statistical modeling and causal inference are performed in four R scripts:

- **Matching-Algorithms.R**  
Propensity score matching and balance diagnostics.
- **PPML-NBR.R**  
Main panel regressions (PPML and Negative Binomial with fixed effects).
- **PPML-NBR-No-Fixed-Effects.R**  
Regressions without fixed effects.
- **PPML-NBR-Interaction.R**  
Interaction models for face/text thumbnail elements.

Please see comments in each script for detailed instructions and input/output file paths.

---

## Tests

- **Unit and integration tests** are not included by default.
- To verify the pipeline:
- Run each script sequentially with a small test dataset.
- Check for expected CSV outputs at each stage.
- For Python, consider using `pytest` for custom test scripts.

---

## Technologies Used

| Language/Framework | Purpose                        |
|--------------------|--------------------------------|
| Python 3.8+        | Data collection & image analysis|
| R 4.2+             | Statistical modeling & plotting |
| pandas, numpy      | Data wrangling                 |
| OpenCV, PIL        | Image processing               |
| DeepFace, YOLO     | Face/object detection          |
| easyOCR            | Text extraction from images    |
| tidyverse, MatchIt | R data analysis & matching     |
| fixest, cobalt     | R regression & diagnostics     |

---

## License

**YouTube API Usage:**  
This project uses the YouTube Data API.
- **You must comply with the YouTube API Services Terms of Service.**
- No redistribution of YouTube audiovisual content is permitted except as allowed by the API terms.
- You retain no rights to YouTube content or data beyond what is expressly granted for API use.
- For academic or non-commercial use, API access is free up to quota limits; commercial use may require additional permissions.


---

## Contact

**Kit Chan**  
k.k.k.chan@tilburguniversity.edu

*For questions, academic collaboration, or issues, please contact via email.*

---

> **Note:**  
> - Replace placeholder paths and filenames as needed for your environment.
> - Always keep your API keys private and secure.
> - For full compliance, regularly review the YouTube API Services Terms of Service.

