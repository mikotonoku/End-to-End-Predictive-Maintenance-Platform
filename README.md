# End-to-End Predictive Maintenance Platform

> **Production-ready Machine Learning system for aircraft engine Remaining Useful Life (RUL) prediction.**

This project demonstrates the development of a complete end-to-end Machine Learning platform for predictive maintenance of aircraft engines using the NASA CMAPSS dataset.

Unlike traditional machine learning projects that focus only on model training, this repository covers the entire engineering pipeline required for deploying an ML model in a production-like environment.

The system automatically validates incoming sensor data, performs preprocessing and feature engineering, loads a trained machine learning model, predicts the Remaining Useful Life (RUL) of an engine, exposes predictions through a REST API, and stores prediction logs for future analysis.

The project is designed to demonstrate practical Machine Learning Engineering skills, including data processing, model deployment, API development, containerization, software engineering best practices, and reproducible project organization.

---

# 📑 Table of Contents

* [1. Project Overview](#1-project-overview)

  * [1.1 Motivation](#11-motivation)
  * [1.2 Problem Statement](#12-problem-statement)
  * [1.3 Project Goals](#13-project-goals)
  * [1.4 Key Features](#14-key-features)

---

## Part I — Theoretical Background

* [2. Predictive Maintenance](#2-predictive-maintenance)

  * [2.1 Maintenance Strategies](#21-maintenance-strategies)
  * [2.2 Industrial Applications](#22-industrial-applications)
  * [2.3 Advantages and Challenges](#23-advantages-and-challenges)

* [3. Remaining Useful Life (RUL)](#3-remaining-useful-life-rul)

  * [3.1 Definition of RUL](#31-definition-of-rul)
  * [3.2 RUL Prediction Problem](#32-rul-prediction-problem)
  * [3.3 Evaluation Metrics](#33-evaluation-metrics)

* [4. NASA CMAPSS Dataset](#4-nasa-cmapss-dataset)

  * [4.1 Dataset Description](#41-dataset-description)
  * [4.2 Sensor Measurements](#42-sensor-measurements)
  * [4.3 Target Variable](#43-target-variable)
  * [4.4 Train/Test Split](#44-traintest-split)

* [5. Machine Learning Background](#5-machine-learning-background)

  * [5.1 Selected Model](#51-selected-model)
  * [5.2 Feature Engineering](#52-feature-engineering)
  * [5.3 Data Validation](#53-data-validation)
  * [5.4 Model Evaluation](#54-model-evaluation)

* [6. Technology Stack](#6-technology-stack)

  * [6.1 Python](#61-python)
  * [6.2 Machine Learning Libraries](#62-machine-learning-libraries)
  * [6.3 FastAPI](#63-fastapi)
  * [6.4 Docker](#64-docker)
  * [6.5 Testing Tools](#65-testing-tools)

---

## Part II — Project Implementation

* [7. System Architecture](#7-system-architecture)

  * [7.1 Overall Architecture](#71-overall-architecture)
  * [7.2 Data Flow](#72-data-flow)
  * [7.3 Project Structure](#73-project-structure)

* [8. Data Pipeline](#8-data-pipeline)

  * [8.1 Input Validation](#81-input-validation)
  * [8.2 Data Preprocessing](#82-data-preprocessing)
  * [8.3 Feature Engineering Pipeline](#83-feature-engineering-pipeline)
  * [8.4 Feature Scaling](#84-feature-scaling)

* [9. Machine Learning Pipeline](#9-machine-learning-pipeline)

  * [9.1 Model Training](#91-model-training)
  * [9.2 Hyperparameter Tuning](#92-hyperparameter-tuning)
  * [9.3 Model Evaluation](#93-model-evaluation)
  * [9.4 Model Serialization](#94-model-serialization)

* [10. Prediction Service](#10-prediction-service)

  * [10.1 Prediction Workflow](#101-prediction-workflow)
  * [10.2 REST API](#102-rest-api)
  * [10.3 Request & Response Examples](#103-request--response-examples)
  * [10.4 Error Handling](#104-error-handling)

* [11. Deployment](#11-deployment)

  * [11.1 Docker](#111-docker)
  * [11.2 Running the Application](#112-running-the-application)
  * [11.3 Environment Configuration](#113-environment-configuration)

* [12. Logging & Monitoring](#12-logging--monitoring)

  * [12.1 Prediction Logging](#121-prediction-logging)
  * [12.2 Error Logging](#122-error-logging)
  * [12.3 Monitoring](#123-monitoring)

* [13. Testing](#13-testing)

  * [13.1 Unit Tests](#131-unit-tests)
  * [13.2 API Tests](#132-api-tests)
  * [13.3 End-to-End Tests](#133-end-to-end-tests)

---

## Part III — Results

* [14. Experimental Results](#14-experimental-results)

  * [14.1 Performance Metrics](#141-performance-metrics)
  * [14.2 Prediction Examples](#142-prediction-examples)
  * [14.3 Error Analysis](#143-error-analysis)

* [15. Future Improvements](#15-future-improvements)

---

## Appendix

* [16. Installation Guide](#16-installation-guide)
* [17. Usage](#17-usage)
* [18. Repository Structure](#18-repository-structure)
* [19. References](#19-references)
* [20. License](#20-license)

---

# 1. Project Overview

## 1.1 Motivation

Modern industrial systems continuously generate large volumes of sensor data during operation. Traditionally, maintenance has been performed either after a component failure (reactive maintenance) or according to predefined service schedules (preventive maintenance). Although these approaches are widely used, they often result in unnecessary maintenance costs, unexpected equipment downtime, and reduced operational efficiency.

Predictive Maintenance has emerged as a data-driven alternative that enables maintenance decisions to be based on the actual condition of equipment rather than fixed schedules. By continuously analyzing sensor measurements, machine learning models can estimate the health state of a system and predict when maintenance should be performed before a failure occurs.

Remaining Useful Life (RUL) prediction is one of the key tasks within predictive maintenance. Accurate estimation of the remaining operational lifetime allows organizations to optimize maintenance schedules, reduce operational costs, improve equipment reliability, and increase safety in critical industries such as aviation, manufacturing, energy production, and transportation.

This project was developed to demonstrate how a complete production-oriented Machine Learning system for RUL prediction can be designed, implemented, and deployed using modern software engineering and MLOps practices.

---

## 1.2 Problem Statement

Many machine learning projects focus exclusively on model development and evaluation while overlooking the engineering components required for deploying models into real-world environments.

In practical industrial applications, prediction is only one stage of a much larger workflow. Incoming data must first be validated, preprocessed, transformed into appropriate features, passed through a trained model, and finally delivered to end users through a reliable inference service. In addition, prediction requests, execution times, and potential errors should be logged for monitoring and future analysis.

The objective of this project is to bridge the gap between machine learning research and production deployment by implementing a complete end-to-end prediction platform capable of automatically processing aircraft engine sensor data and estimating the Remaining Useful Life of each engine.

---

## 1.3 Project Goals

The main objectives of this project are:

* Develop a complete end-to-end Machine Learning pipeline for Remaining Useful Life prediction.
* Implement automated validation of incoming sensor data.
* Perform preprocessing and feature engineering before model inference.
* Train and evaluate a regression model using the NASA CMAPSS dataset.
* Build a REST API using FastAPI for serving predictions.
* Containerize the application using Docker.
* Implement prediction logging and error handling.
* Organize the project following software engineering best practices.
* Produce comprehensive technical documentation describing both theoretical concepts and implementation details.

---

## 1.4 Key Features

The project includes the following major components:

* End-to-end Machine Learning pipeline.
* Remaining Useful Life prediction for aircraft engines.
* Automated data validation.
* Data preprocessing and feature engineering.
* Machine Learning model inference.
* REST API developed with FastAPI.
* Docker-based deployment.
* Prediction logging and monitoring.
* Modular and scalable project architecture.
* Reproducible experiments and technical documentation.

---

# 2. Predictive Maintenance

Predictive Maintenance (PdM) is a maintenance strategy that uses historical and real-time operational data to estimate the health condition of industrial equipment and predict potential failures before they occur. Instead of relying on fixed maintenance schedules or waiting until equipment breaks down, predictive maintenance continuously monitors system performance and determines the optimal time for maintenance interventions based on the actual condition of the asset [[1]](#ref-1) [[2]](#ref-2).

Recent advances in machine learning, artificial intelligence, Industrial Internet of Things (IIoT), and cloud computing have significantly expanded the capabilities of predictive maintenance systems. Modern industrial environments continuously collect large amounts of sensor data, including temperature, pressure, vibration, rotational speed, fuel consumption, and other operational measurements. Machine learning models analyze these data streams to identify degradation patterns and estimate the Remaining Useful Life (RUL) of equipment, enabling organizations to schedule maintenance before critical failures occur [[1]](#ref-1) [[3]](#ref-3).

Predictive maintenance has become one of the most important applications of artificial intelligence in Industry 4.0, where reducing downtime, improving operational efficiency, and increasing equipment reliability are critical business objectives [[1]](#ref-1) [[3]](#ref-3).

---

## 2.1 Maintenance Strategies

Industrial maintenance strategies can generally be divided into three major categories.

### Reactive Maintenance

Reactive maintenance, often referred to as **run-to-failure maintenance**, involves repairing or replacing equipment only after a failure has occurred. Although this strategy minimizes maintenance activities during normal operation, unexpected failures often lead to costly production downtime, emergency repairs, safety risks, and secondary damage to other system components [[1]](#ref-1) [[2]](#ref-2).

### Preventive Maintenance

Preventive maintenance performs inspections and component replacement at predefined time intervals regardless of the actual equipment condition. While this approach reduces the probability of catastrophic failures, it frequently results in unnecessary maintenance operations, increased labor costs, and premature replacement of components that still have significant remaining service life [[1]](#ref-1) [[2]](#ref-2).

### Predictive Maintenance

Predictive maintenance combines continuous condition monitoring with data analytics and machine learning algorithms to estimate equipment health in real time. Maintenance actions are scheduled only when degradation indicators suggest that a failure is likely to occur in the near future. This approach minimizes unnecessary maintenance while substantially reducing unexpected failures and operational downtime [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3).

---

## 2.2 Industrial Applications

Predictive maintenance is widely adopted across industries where equipment failures result in significant financial losses or safety risks.

### Aviation

Aircraft engines contain hundreds of components operating under extreme temperatures and mechanical stress. Continuous monitoring of engine sensors enables machine learning models to estimate Remaining Useful Life (RUL), allowing maintenance to be planned before critical failures occur. The NASA CMAPSS dataset used in this project represents one of the most widely used benchmark datasets for this application.

### Manufacturing

Manufacturing facilities employ predictive maintenance to monitor production machinery, conveyor systems, pumps, compressors, and robotic equipment. Early detection of abnormal operating conditions reduces production interruptions and improves overall equipment effectiveness (OEE) [[1]](#ref-1).

### Energy Industry

Power plants, wind turbines, gas turbines, and electrical generators continuously generate operational data that can be analyzed to detect degradation before failures occur. Predictive maintenance improves equipment availability while reducing maintenance costs [[1]](#ref-1) [[3]](#ref-3).

### Transportation

Railway systems, heavy-duty vehicles, mining equipment, and marine engines benefit from predictive maintenance by reducing unexpected breakdowns and optimizing maintenance schedules according to equipment condition rather than fixed intervals [[2]](#ref-2).

---

## 2.3 Advantages and Challenges

### Advantages

Predictive maintenance provides numerous advantages over traditional maintenance strategies:

- reduced unplanned equipment downtime;
- lower maintenance costs;
- increased equipment reliability;
- longer asset lifetime;
- improved operational efficiency;
- optimized spare parts management;
- enhanced workplace safety;
- data-driven maintenance planning [[1]](#ref-1) [[2]](#ref-2) [[3]](#ref-3).

### Challenges

Despite its advantages, predictive maintenance also presents several technical challenges:

- large volumes of sensor data require efficient storage and processing;
- missing values and noisy sensor measurements may reduce model performance;
- obtaining failure data is often difficult because equipment failures are relatively rare;
- operating conditions may change over time, resulting in distribution shifts between training and production data;
- industrial systems often require real-time prediction with low latency and high reliability;
- successful deployment requires integration with existing production systems and maintenance workflows [[1]](#ref-1) [[3]](#ref-3).

Understanding these challenges is essential for designing robust machine learning systems capable of operating reliably in real industrial environments.

---

# 3. Remaining Useful Life (RUL)

Remaining Useful Life (RUL) refers to the estimated amount of time or number of operating cycles that a system or component can continue functioning before it reaches the end of its useful life or experiences failure. RUL estimation is one of the fundamental tasks of predictive maintenance because it enables maintenance activities to be scheduled according to the actual health condition of equipment rather than predefined maintenance intervals [[4]](#ref-4) [[5]](#ref-5).

Unlike fault detection, which determines whether a failure has already occurred, RUL prediction estimates how much useful operating life remains before the failure is expected to happen. This information allows organizations to optimize maintenance planning, minimize unexpected downtime, reduce maintenance costs, and improve equipment reliability [[4]](#ref-4) [[6]](#ref-6).

Modern RUL prediction systems rely on continuous streams of sensor measurements collected during equipment operation. Machine learning algorithms analyze historical degradation patterns and estimate future system degradation, producing a prediction of the remaining operational lifetime [[5]](#ref-5) [[7]](#ref-7).

---

## 3.1 Definition of RUL

Remaining Useful Life represents the interval between the current operating condition of a system and the point at which the system can no longer perform its intended function according to predefined performance criteria [[4]](#ref-4).

Depending on the application, RUL may be expressed as:

- remaining operating cycles;
- remaining operating hours;
- remaining mileage;
- remaining production time.

For aircraft engines, including the NASA CMAPSS dataset used in this project, RUL is measured as the number of remaining operational cycles before engine failure [[7]](#ref-7).

Accurate estimation of RUL enables maintenance engineers to replace or repair components before catastrophic failures occur while maximizing the utilization of expensive industrial assets [[5]](#ref-5).

---

## 3.2 RUL Prediction Problem

From a machine learning perspective, Remaining Useful Life prediction is formulated as a **supervised regression problem**.

Each observation consists of multiple sensor measurements describing the current operating state of the equipment:

```text
x = [sensor₁, sensor₂, ..., sensorₙ]
````

The learning objective is to estimate a continuous target variable:

```text
ŷ = Remaining Useful Life
```

Unlike classification problems that predict discrete categories such as *healthy* or *faulty*, regression models produce continuous numerical predictions representing the expected remaining lifetime of the equipment [[4]](#ref-4).

Several challenges make RUL prediction particularly difficult:

* equipment degradation is typically nonlinear;
* operating conditions may vary over time;
* sensor measurements often contain noise;
* failures occur relatively rarely, limiting the amount of labeled degradation data;
* different assets may degrade at different rates despite operating under similar conditions [[4]](#ref-4) [[5]](#ref-5).

To address these challenges, modern RUL prediction systems employ feature engineering, data preprocessing, and advanced machine learning algorithms capable of learning complex degradation patterns from historical sensor data [[5]](#ref-5) [[7]](#ref-7).

---

## 3.3 Evaluation Metrics

Since Remaining Useful Life prediction is a regression task, model performance is evaluated using regression metrics that quantify the difference between predicted and actual RUL values.

The most commonly used evaluation metrics include:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and true RUL values.

Lower MAE indicates more accurate predictions.

---

### Root Mean Squared Error (RMSE)

RMSE calculates the square root of the average squared prediction error.

Because larger errors receive greater penalties, RMSE is particularly useful when large prediction errors are undesirable.

The NASA CMAPSS benchmark commonly reports RMSE as one of its primary evaluation metrics [[7]](#ref-7).

---

### Coefficient of Determination (R²)

The coefficient of determination measures how well the predicted values explain the variance of the true RUL values.

An R² value close to 1 indicates excellent predictive performance, while values closer to 0 indicate poor explanatory power.

---

### NASA Scoring Function

In addition to traditional regression metrics, the NASA CMAPSS benchmark frequently employs a specialized scoring function that penalizes late predictions more heavily than early predictions because unexpected failures generally have more severe operational consequences [[4]](#ref-4) [[7]](#ref-7).

Using multiple evaluation metrics provides a more comprehensive assessment of model performance since each metric captures different characteristics of prediction quality.

---

# 4. NASA CMAPSS Dataset 

The experiments presented in this project are based on the **Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset developed by the NASA Prognostics Center of Excellence (PCoE). The dataset is one of the most widely used public benchmarks for Remaining Useful Life (RUL) prediction and predictive maintenance research because it realistically simulates the degradation process of aircraft turbofan engines under various operating conditions [[8]](#ref-8) [[9]](#ref-9). 

Unlike real industrial datasets, CMAPSS is generated using a high-fidelity engine simulator that models progressive engine degradation until system failure. This approach allows researchers to evaluate machine learning algorithms using fully labeled degradation trajectories while avoiding the confidentiality issues associated with real aircraft engine data [[8]](#ref-8). 

The dataset has become the standard benchmark for evaluating machine learning methods for prognostics and health management (PHM), and it is extensively used in both academic research and industrial applications [[8]](#ref-8). 

--- 

## 4.1 Dataset Description 

The CMAPSS dataset contains multivariate time-series data collected from multiple simulated aircraft turbofan engines operating under different environmental and operating conditions. 

Each engine begins operation in a healthy state and gradually degrades over time until failure occurs. Every engine follows a unique degradation trajectory, meaning that engines fail after different numbers of operational cycles. 

Each observation within the dataset represents a single operational cycle and contains: 

- engine identifier;
- operational cycle number;
- three operational setting variables;
- twenty-one sensor measurements.

The dataset consists of four independent subsets (FD001–FD004), each representing different combinations of operating conditions and fault modes [[8]](#ref-8). 

--- 

## 4.2 Sensor Measurements 

For every operational cycle, the simulator records measurements from multiple virtual sensors describing the current health state of the engine. 

The available variables include: 

- **Engine ID** — unique engine identifier;
- **Cycle** — current operating cycle;
- **Operational Settings (3 variables)** — environmental and operating conditions;
- **Sensor Measurements (21 variables)** — simulated engine sensor readings.

The sensor values represent measurements such as pressures, temperatures, rotational speeds, fuel flow characteristics, and other internal engine parameters. Although some sensors remain nearly constant throughout engine operation, others exhibit clear degradation trends that are highly informative for Remaining Useful Life prediction [[8]](#ref-8). 

In practical machine learning applications, feature engineering and feature selection are commonly applied to identify the most informative sensor signals before model training. 

--- 

## 4.3 Target Variable 

The prediction target is the **Remaining Useful Life (RUL)** of each engine. 

For every observation, RUL represents the number of operating cycles remaining before engine failure. 

During training, the complete degradation trajectory is available, making it possible to calculate the exact RUL value for every cycle. During testing, only partial engine trajectories are provided, and the objective is to predict the remaining lifetime at the last available observation of each engine [[8]](#ref-8). 

This formulation makes the problem a supervised regression task where the input consists of multivariate sensor measurements and the output is a continuous numerical estimate of the remaining operating cycles. 

--- 

## 4.4 Train/Test Split 

The CMAPSS dataset provides separate training and testing datasets. 

The training dataset contains complete degradation trajectories for each engine, ending at the failure point. Consequently, the exact Remaining Useful Life can be computed for every operational cycle. 

The testing dataset contains truncated engine trajectories that terminate before failure occurs. Instead of providing the remaining lifetime directly, NASA supplies a separate file containing the true RUL value for the final observation of every engine. This setup closely resembles real industrial scenarios where equipment has not yet failed and the objective is to estimate its remaining operational lifetime [[8]](#ref-8). 

The dataset used in this project was downloaded from Kaggle, while the original dataset was developed and published by the NASA Prognostics Center of Excellence [[8]](#ref-8) [[10]](#ref-10). 

---

## References

<a id="ref-1"></a>

**[1]** IBM. *What is Predictive Maintenance?* Available: https://www.ibm.com/think/topics/predictive-maintenance

<a id="ref-2"></a>

**[2]** ManWinWin Software. *Predictive Maintenance: Complete Guide.* Available: https://www.manwinwin.com/predictive-maintenance/

<a id="ref-3"></a>

**[3]** Neural Concept. *How AI is Used in Predictive Maintenance.* Available: https://www.neuralconcept.com/post/how-ai-is-used-in-predictive-maintenance

<a id="ref-4"></a>

**[4]** Elsheikh A., Yacout S., Ouali M.-S. *Bidirectional Handshaking Between Maintenance and Product Design for Remaining Useful Life Prediction.* Procedia CIRP, 2014. Available: https://www.sciencedirect.com/science/article/pii/S2212827114001140

<a id="ref-5"></a>

**[5]** MathWorks. *Three Ways to Estimate Remaining Useful Life for Predictive Maintenance.* Available: https://www.mathworks.com/company/technical-articles/three-ways-to-estimate-remaining-useful-life-for-predictive-maintenance.html

<a id="ref-6"></a>

**[6]** Advanced Technology Services. *What Is Remaining Useful Life (RUL)?* Available: https://www.advancedtech.com/blog/what-is-remaining-useful-life-rul/

<a id="ref-7"></a>

**[7]** Kaggle. *Remaining Useful Lifetime Prediction.* Available: https://www.kaggle.com/code/sasakitetsuya/remaining-useful-lifetime-prediction

<a id="ref-8"></a> 

**[8]** NASA. *CMAPSS Jet Engine Simulated Data.* Available: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

<a id="ref-9"></a> 

**[9]** NASA. *Prognostics Center of Excellence Data Set Repository.* Available: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

<a id="ref-10"></a> 

**[10]** Behrad Babaeian. *NASA CMAPSS Dataset.* Kaggle. Available: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
