# AcTBeCalf Dataset Summary

*This is an AI-generated summary of the AcTBeCalf dataset and associated code repository.*

## Dataset Description

The **AcTBeCalf dataset** is a comprehensive dataset designed to support the classification of pre-weaned calf behaviors from accelerometer data. It contains detailed accelerometer readings aligned with annotated behaviors, providing a valuable resource for research in multivariate time-series classification and animal behavior analysis.

### Data Collection

The dataset includes accelerometer data collected from **30 pre-weaned Holstein Friesian and Jersey calves**, housed in group pens at the **Teagasc Moorepark Research Farm, Ireland**. Each calf was equipped with a 3D accelerometer sensor (AX3, Axivity Ltd, Newcastle, UK) sampling at **25 Hz** and attached to a neck collar from one week of birth over 13 weeks.

### Dataset Characteristics

- **Duration**: 27.4 hours of accelerometer data aligned with calf behaviors
- **Behaviors**: Includes both prominent behaviors (lying, standing, running) and less frequent behaviors (grooming, social interaction, abnormal behaviors)
- **Format**: Single CSV file with timestamped accelerometer readings and behavior annotations

### Data Structure

The dataset consists of a single CSV file with the following columns:

- **`dateTime`**: Timestamp of the accelerometer reading, sampled at 25 Hz
- **`calfid`**: Identification number of the calf (1-30)
- **`accX`**: Accelerometer reading for the X axis (top-bottom direction)*
- **`accY`**: Accelerometer reading for the Y axis (backward-forward direction)*
- **`accZ`**: Accelerometer reading for the Z axis (left-right direction)*
- **`behavior`**: Annotated behavior based on an ethogram of 23 behaviors
- **`segId`**: Segment identification number representing all readings of the same behavior segment

*\* Directions are mentioned in relation to the position of the accelerometer sensor on the calf.*

## Code Repository Structure

The dataset is accompanied by a comprehensive codebase that implements a complete machine learning pipeline from raw data preprocessing to behavior prediction models. The code is organized into three main categories:

### Data Preprocessing Pipeline

1. **`1_accelerometer_time_correction.ipynb`**
   - Corrects accelerometer time drift using shake pattern detection
   - Ensures alignment of accelerometer data with reference time
   - Implements magnitude calculation from 3-axis accelerometer data

2. **`2_aligning_accelerometer_data_with_annotations.ipynb`**
   - Aligns accelerometer time series with annotated behaviors based on timestamps
   - Synchronizes sensor data with behavioral annotations from BORIS video analysis

3. **`3_manual_inspection_ts_validation.ipynb`**
   - Provides manual inspection process for ensuring accurate alignment
   - Quality control validation of accelerometer data with annotated behaviors

4. **`4_additional_ts_generation.ipynb`**
   - Generates additional time-series features from original X, Y, Z accelerometer readings
   - Creates derived features: Magnitude, ODBA, VeDBA, pitch, and roll

### Core Library (`holsteinlib/`)

A custom Python library providing essential utilities:

- **`shake_pattern_detector.py`**: Algorithm to detect shake patterns in accelerometer signals for time alignment
- **`feature_functions.py`**: Comprehensive feature extraction including:
  - Accelerometer-specific features (magnitude, VeDBA, ODBA, pitch, roll)
  - Statistical features (mean, median, std, entropy, motion variation, kurtosis, skew)
  - Signal processing for static and dynamic components
- **`windowing.py`**: Time-series windowing with configurable parameters
- **`functions.py`**: Data manipulation and utility functions
- **`genSplit.py`**: Generalized subject separation logic for ML model training, validation, and testing

### Machine Learning Models

1. **`active_inactive_classification.ipynb`**
   - Binary classification of behaviors into active vs inactive categories
   - Uses RandomForest model with hand-crafted statistical features
   - **Performance**: 92% balanced accuracy

2. **`four_behaviour_classification.ipynb`**
   - Multi-class classification into four categories: drinking milk, lying, running, and other
   - Employs MiniRocket feature derivation mechanism with RidgeClassifierCV
   - **Performance**: 84% balanced accuracy

## Technical Approach

### Data Processing Workflow

1. **Raw accelerometer data** → Time correction using shake patterns
2. **Corrected data** → Alignment with behavioral video annotations
3. **Aligned data** → Quality validation and feature engineering
4. **Processed data** → Window-based feature extraction
5. **Features** → Machine learning model training and evaluation

### Key Technical Features

- **Multi-modal approach**: Combines traditional statistical features with modern time series methods (MiniRocket)
- **Robust preprocessing**: Handles time synchronization challenges between sensors and video
- **Domain expertise**: Uses cattle-specific behavioral knowledge for feature engineering
- **Production-ready**: Modular design with reusable library components

## Citations

When using this dataset, please cite one of the following papers:

1. Dissanayake, O., McPherson, S. E., Allyndrée, J., Kennedy, E., Cunningham, P., & Riaboff, L. (2024). Evaluating ROCKET and Catch22 features for calf behaviour classification from accelerometer data using Machine Learning models. *arXiv preprint arXiv:2404.18159*.

2. Dissanayake, O., McPherson, S. E., Allyndrée, J., Kennedy, E., Cunningham, P., & Riaboff, L. (2024). Development of a digital tool for monitoring the behaviour of pre-weaned calves using accelerometer neck-collars. *arXiv preprint arXiv:2406.17352*.

## Research Applications

This dataset is particularly valuable for:
- Animal behavior recognition and classification
- Time series analysis and machine learning research
- Agricultural technology development
- Livestock monitoring system development
- Multivariate sensor data analysis methodologies

---

*Dataset source: Zenodo repository*  
*Code analysis generated by AI assistant on September 26, 2025*