# DiabTrack-Diabetes-Predictor
A clinical decision-support system that predicts diabetes progression through ensemble machine learning. The platform integrates NHANES population health data with PIMA Indian diabetes records, harmonizing 10,000+ patient profiles into a unified analytical framework.

Core architecture employs a weighted voting ensemble: Random Forest (400 trees) paired with two XGBoost variants (300/200 trees) at 1.3:1.0:0.7 ratios. This configuration achieves 90.2% accuracy by leveraging complementary learning approaches—XGBoost captures complex patterns while Random Forest ensures stability.

Twelve clinical parameters drive predictions: glucose levels, HbA1c, age, BMI, calculated HOMA-IR, waist circumference, systolic blood pressure, insulin levels, menstrual history, gender, and family diabetes history. Advanced preprocessing handles medical data complexities through skewness-based imputation and IQR outlier management.

Deployed via Streamlit with medical-grade interface design, the system provides real-time risk assessment with 35% optimized threshold balancing clinical sensitivity (91.2%) against specificity. Complete pipeline demonstrates production ML implementation from raw data processing through model serialization to web deployment.

The solution addresses pre-diabetes progression prediction—identifying high-risk patients who require intervention beyond standard glucose monitoring. Technical implementation showcases full-stack data science capabilities with healthcare domain specialization, emphasizing both predictive performance and clinical usability.

