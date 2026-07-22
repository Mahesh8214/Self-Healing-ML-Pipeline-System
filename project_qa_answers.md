# 🧠 Self-Healing ML Pipeline — All Questions Answered

> **Style**: Simple language + Hinglish + examples. Answers based 100% on the actual project code.

---

# LEVEL 1 — Project Overview & Motivation

---

### Q1: Walk me through what your self-healing ML pipeline does in 2 minutes.

**Answer:**
Samjho — ek ML model banaya diamond price predict karne ke liye. Lekin problem yeh hai ki real world mein data time ke saath badal jaata hai. Jab data badalta hai, model ki accuracy girne lagti hai. Normally koi manually check karta tha aur model retrain karta tha. Mera system yeh kaam **automatically** karta hai.

**Key Points:**
- Production mein aane wale data ko **monitor** karta hai batch by batch
- Agar data ki distribution badal gayi (drift detected) AND model ki performance girr gayi → **khud retrain ho jaata hai**
- Naya model better hai toh deploy, worse hai toh purana wala rakhta hai (rollback)
- Sab kuch **log** hota hai: kab drift hua, R² score kya tha, retrain hua ya nahi

**Follow-up: Business problem?**
Companies ko lagata hai unka model theek hai, lekin quietly accuracy gir rahi hoti hai months ke baad. Isse unhe pata hi nahi chalta. Mera system isse **automatically pakadta hai aur fix karta hai** — engineer ki zaroorat nahi.

**Follow-up: Schedule se alag kaise?**
Schedule-based retraining mein hum **blindly** retrain karte hain chahe zaroorat ho ya na ho. Mera system **sirf tab retrain karta hai jab scientific evidence hota hai** (drift + performance drop). Yeh smarter, cheaper aur safer hai.

---

### Q2: "Data drift" kya hota hai simple terms mein?

**Answer:**
Samjho tumne model ko 2020 ke data par train kiya. 2023 mein real world ka data aata hai jo thoda alag hai — diamond sizes bade ho gaye, prices changed, market shift hua. Yahi data drift hai — **training data aur production data ka alag ho jaana**.

**Key Points:**
- Model ne jo patterns seekhe the, woh ab production data mein nahi milte
- Model wahi predictions karta hai jo purane patterns ke basis par hain — isliye accuracy girti hai

**Example/Analogy:**
Doctor ne 1990 ke X-ray data par AI train kiya. Ab 2024 mein newer MRI machines alag type ke scans deti hain. AI ko 1990 wala data yaad hai, 2024 wala samajh nahi aata → yahi drift hai.

**Follow-up: Real-world drift example?**
COVID ke time diamond ki demand ek dum girr gayi, prices crash hua. Pehle ke model ne yeh anticipate nahi kiya tha → drift.

**Follow-up: Roz retrain kyun nahi?**
- Bahut **expensive** hai (compute cost)
- Training data collect karne mein time lagta hai
- Bekar retrain karna model ko **overfit** bhi kar sakta hai
- **Scientific reason** hona chahiye retrain karne ka

---

### Q3: Kya yeh solo project tha ya team?

**Answer:**
Yeh **solo project** hai — maine pura system khud banaya:  
pipeline architecture, drift detection logic, model registry, Streamlit dashboard — sab ek ek cheez maine implement ki.

**Key Points:**
- **Data Ingestion** — khud likha
- **Drift Detector** — KS test + PSI dono khud implement kiya (library use nahi ki)
- **Model Registry** — MLflow ke bina khud JSON-based versioning system banaya
- **Monitoring Pipeline** — complete self-healing loop khud design kiya

---

### Q4: MLflow ya Evidently AI kyun nahi use kiya?

**Answer:**
Inhe use karna easy hota, lekin tab mujhe **andar ki cheezein samajh nahi aatein**. Khud banake maine seekha ki:
- KS test mathematically kaise kaam karta hai
- PSI formula mein log kyun aata hai
- Model versioning ka edge case kya hoti hai

**Key Points:**
- Library use karte toh ek "black box" use kar raha hota
- Self-build se **deep understanding** aaya
- Interview mein "yeh library use ki" kehna weak lagta hai vs "maine yeh implement kiya"

**Follow-up: Kya seekha?**
- Statistical tests ki limitations
- JSON-based registry mein atomic write ka problem
- Circular import ka issue aur `lazy import` pattern

---

### Q5: Kaunsa model monitor kar rahe ho? Regression ya Classification?

**Answer:**
Main **regression model** monitor kar raha hoon — diamond price predict karta hai (continuous number like ₹5,432).

**Follow-up: Diamond price specifically kyun?**
Gemstone dataset publicly available tha, acha real-world regression problem tha jisme drift simulate karna easy tha.

**Follow-up: Classification ke liye bhi kaam karega?**
Haan, **mostly kaam karega** lekin kuch changes chahiye:
- Performance metric: R² → Accuracy/F1/AUC
- PSI: continuous features ke liye hai; categorical drift ke liye Chi-square test better
- Retraining threshold: `score < 0.80` → `f1 < 0.75` etc.

---

# LEVEL 2 — Core Technical Questions

---

### Q6: Production data aane se model retrain tak ka complete data flow batao.

**Answer:**

```
batch_N.csv aaya
    ↓
DataValidation → schema check + missing values check
    ↓ (pass kiya)
DriftDetector → KS test + PSI har numerical feature par
    ↓ (drift_report.json save hua)
PerformanceMonitor → latest model load, batch par predict, R² calculate
    ↓
Decision: drift=True AND R² < 0.80?
    YES → run_training_pipeline() → nayi model train → deploy
    NO  → skip, log karo
    ↓
monitoring_log.json mein entry → batch_log.json mein mark as processed
```

**Follow-up: Intermediate results kahan store hote hain?**
- `artifacts/reports/drift_report.json` — drift results
- `artifacts/monitoring/monitoring_log.json` — har batch ka record
- `artifacts/monitoring/batch_log.json` — processed batches list
- `artifacts/preprocessor.pkl` — fitted transformer
- `artifacts/models/model_vN.pkl` — trained models

**Follow-up: Pipeline crash ho gayi beech mein toh?**
- `batch_log.json` track karta hai kaunse batches complete hue
- Restart par already processed batches **skip** ho jaati hain (idempotent design)
- Lekin agar crash training ke beech mein hua toh incomplete model nahi save hoga (pkl write atomic hai)

---

### Q7: Production batches kaise banaye?

**Answer:**
Project mein yeh **simulated** batches hain — `notebooks/test_drift_data_maker.py` ne 50 CSV files banaye `data/production_batches/` mein. Real production mein yeh batches har kuch ghante mein database se aate.

**Key Points:**
- Humne reference data mein thoda noise add karke batches banaye taaki kuch batches mein drift simulate ho
- Batch 10, 16, 17, 19, 22, 24, 44, 46 mein drift tha (real log se confirmed)

**Follow-up: Batch size aur frequency kaise decide karo?**
- Enough data hona chahiye statistical test ke liye (minimum ~500 rows recommended)
- Frequency business need par depend: e-commerce → hourly, healthcare → daily

**Follow-up: Irregular batch sizes?**
- Not implemented in project, but ideally: minimum rows ka check lagao validation mein
- Agar batch chhota hai → skip karo ya accumulate karo

---

### Q8: monitoring_pipeline.py step by step samjhao.

**Answer:**
```python
# Step 0: Saari batch files lo aur sort karo
batches = sorted(os.listdir("data/production_batches"))

for batch_file in batches:
    # Step 1: Pehle already process hua? Skip karo
    if is_batch_processed(batch_file): continue

    # Step 2: Data valid hai? Schema ok? Missing values?
    status = DataValidation().initiate_data_validation(batch_path)
    if not status: continue  # Invalid batch skip

    # Step 3: Drift detect karo (KS + PSI)
    drift = DriftDetector().initiate_drift_detection(reference, batch)

    # Step 4: Current model ki performance check karo
    score = PerformanceMonitor().evaluate_model(batch_path)

    # Step 5: Retrain decision
    if drift and score < 0.8:
        run_training_pipeline()

    # Step 6: Log karo + mark as processed
    log_monitoring(batch, drift, score, retrained)
    mark_batch_processed(batch_file)
```

**Follow-up: Kaunsa model version use hota hai comparison ke liye?**
`ModelRegistry().get_latest_model()` → jo bhi `model_registry.json` mein `"latest_model"` field hai, wahi use hota hai.

**Follow-up: Production data nahi hai toh?**
Pipeline `listdir()` karta hai — agar folder empty hai toh loop hi nahi chalta, no error.

---

### Q9: Training, Monitoring, Prediction alag kyun rakhe?

**Answer:**
Yeh **Separation of Concerns** design principle hai — har cheez apna kaam kare.

**Key Points:**
- **Training pipeline** → data se model banana
- **Monitoring pipeline** → deployment ke baad dekhna kya ho raha hai
- **Prediction pipeline** → user ka input leke answer dena

Inka kaam alag hai, agar ek fail ho toh baaki dono independent chalta rahega.

**Follow-up: Overhead?**
Thoda extra code hai, lekin yeh **maintainable** aur **debuggable** hai. Ek badi file mein sab hota toh ek bug sab tod deta.

**Follow-up: Communicate kaise karte hain?**
- Shared files ke through: `preprocessor.pkl`, `model_registry.json`, `monitoring_log.json`
- Monitoring pipeline internally `run_training_pipeline()` ko call karta hai (lazy import)

---

# Drift Detection Methodology

---

### Q10: KS test kyun choose kiya drift detection ke liye?

**Answer:**
KS test (Kolmogorov-Smirnov) ek powerful statistical test hai jo **poori distribution** compare karta hai — sirf mean ya variance nahi.

**Example/Analogy:**
Socho reference data mein diamonds mostly 0.5-1.0 carat ke hain. Production mein suddenly 2.0-3.0 carat ke aane lage. Mean same ho sakta hai, lekin **shape** alag hai. KS test yeh pakad leta hai, mean comparison nahi pakad sakta.

**Follow-up: Limitations?**
- Only **continuous** (numerical) features ke liye kaam karta hai
- **Large datasets** mein chhoti bhi distribution difference detect hoti hai (false positives)
- Alag distribution shapes mein sometimes miss karta hai

**Follow-up: Sirf mean ya variance kyun nahi compare karte?**
Mean: `[1,1,1,9,9,9]` aur `[5,5,5,5,5,5]` dono ka mean = 5, lekin distributions **bilkul alag** hain. KS test yeh pakad leta.

---

### Q11: KS test + PSI dono kyun?

**Answer:**
Dono alag cheez measure karte hain — ek dusre ke complement hain.

| | KS Test | PSI |
|---|---|---|
| **Kya measure karta hai** | Kya drift hai ya nahi (yes/no) | Kitna bada drift hai (magnitude) |
| **Output** | p-value (probability) | Score (0 = no drift, >0.2 = drift) |
| **Acha hai** | Statistical significance ke liye | Business impact ke liye |

**Follow-up: Conflicting signals ho sakte hain?**
Haan! KS p=0.04 (drift!) lekin PSI=0.05 (low magnitude). Matlab: statistically alag hai lekin practically chhhoti baat hai.

**Follow-up: Conflict mein kise trust karein?**
Project mein: **OR logic** — agar koi bhi ek fires toh drift flag. Conservative approach (better safe than sorry). Production mein PSI zyada actionable hota hai.

---

### Q12: PSI formula mein log(0) ka problem?

**Answer:**
PSI formula: `PSI = (Actual% - Expected%) * log(Actual%/Expected%)`

Agar kisi bin mein 0 observations hain toh `log(0/something)` = `log(0)` = **negative infinity** → error!

**Key Points:**
- Yeh divide by zero equivalent hai

**Follow-up: Code mein kaise handle kiya?**
```python
if e == 0:
    e = 0.0001   # ← Epsilon value
if a == 0:
    a = 0.0001   # ← Dodge the zero
```
Zero ko ek chhoti value se replace kiya — math main issue nahi aata.

**Follow-up: Testing mein encounter hua?**
Haan — jab production batch mein kuch features ke extreme values nahi the, kuch bins empty ho gaye. Isliye yeh fix implement kiya.

---

### Q13: Feature-level drift ko system-level decision mein kaise aggregate karte ho?

**Answer:**
Simple OR logic: **koi bhi ek feature drift kare toh poora batch drifted maana jaata.**

```python
drift_detected = False
for col in numerical_columns:
    if p_value < 0.05 OR psi_score > 0.2:
        drift_results[col]["drift_detected"] = True
        drift_detected = True  # ← Ek bhi feature drift kare toh True
```

**Follow-up: Sirf ek feature drift kare toh?**
Is project mein: haan, tab bhi drift flag set hoga. Conservative approach.

**Follow-up: Sab features equal importance?**
Is project mein haan, sab equal. **Not implemented but ideally**: feature importance weights use karo. `carat` zyada important hai price ke liye → uska drift zyada weight milna chahiye.

---

### Q14: KS test mein p-value threshold 0.05 kyun?

**Answer:**
0.05 ek **standard scientific convention** hai — matlab "5% se kam probability hai ki yeh difference random chance se hua". Statistics mein yah alpha level widely accepted hai.

**Follow-up: System kitna sensitive hai?**
- p-value threshold kam karo (e.g., 0.01) → kam false alarms, lekin real drift miss ho sakta hai
- Zyada karo (e.g., 0.10) → zyada alerts, lekin false positives bhi

**Follow-up: Tune kiya experimentally?**
Standard 0.05 use kiya — project mein no tuning. Ideally production mein validation data par tune karna chahiye.

---

# Model Registry & Versioning

---

### Q15: Model Registry versions kaise track karta hai?

**Answer:**
Ek simple `model_registry.json` file hai — ye ek "ledger" ki tarah kaam karta hai.

```json
{
  "latest_model": "artifacts/models/model_v31.pkl",
  "versions": [
    {
      "version": "v1",
      "model_path": "artifacts/models/model_v1.pkl",
      "timestamp": "2026-03-09 18:07:05",
      "reason": "manual_training"
    }
    ...
  ]
}
```

**Follow-up: Kya metadata store hota hai?**
- Version number (v1, v2, ...)
- Model file path
- Timestamp (kab train hua)
- Reason (manual_training / performance_degradation_after_drift)

**Follow-up: Rollback possible hai?**
Manually haan — `model_registry.json` mein `"latest_model"` change karo aur file path pehle wale version ka daal do. **Automatic rollback not implemented** — but `model_trainer.py` mein rollback logic hai: agar naya model worse hai toh deploy hi nahi karta.

---

### Q16: Naya model deploy karein ya purana rakhein — decision kaise?

**Answer:**
Simple R² comparison:

```python
new_score = r2_score(y_test, new_model.predict(X_test))
old_score = r2_score(y_test, old_model.predict(X_test))

if new_score > old_score:
    DEPLOY new model
else:
    KEEP old model (rollback)
```

**Follow-up: Naya model validation par worse nikle toh?**
Deploy hi nahi hota — `deploy_new_model = False` flag set hota hai. Purana model production mein rehta hai.

**Follow-up: A/B Testing?**
**Not implemented in project.** Ideally: 10% traffic naye model ko, 90% purane ko → compare → winner deploy.

---

### Q17: Model artifacts physically kahan store hote hain?

**Answer:**
`artifacts/models/model_v1.pkl` se `model_v31.pkl` — local disk par.
- File size: ~14-18 MB per model
- 31 models = ~450 MB total disk usage

**Follow-up: Storage full ho gaya toh?**
**Not handled in project.** Ideally: last N versions rakho, purane delete karo.

**Follow-up: Cleanup strategy?**
Not implemented. Best practice: S3/GCS mein store karo with lifecycle policies — 90 din se purane models auto-delete.

---

# Retraining Logic

---

### Q18: Drift AND Performance drop — dono kyun? Sirf drift kyun nahi?

**Answer:**
Yeh ek bahut smart design decision hai.

**Samjho yeh scenarios:**
1. **Drift hai, performance nahi giri** → Model drift ke bawajood adapt ho gaya. Retrain ki zaroorat nahi!
2. **Performance giri, drift nahi** → Shayad koi aur issue hai (data quality, outliers). Retrain karna risky.
3. **Drift + Performance drop** → Clear signal ki model outdated ho gaya. **Retrain karo!**

**Key Points:**
- Sirf drift par retrain karna = unnecessary retraining = waste of compute
- Dono conditions AND honi chahiye → confident decision

**Follow-up: Drift bina performance drop ke?**
Haan possible — model robust hota hai agar drift mild ho ya correlated features predict karne mein help karein.

**Follow-up: Performance drop bina drift ke?**
Shayad data mein outliers aaye, ya batch chhota tha isliye R² zyada variable tha. Drift test additional confirmation deta hai.

---

### Q19: Production mein ground truth labels nahi hote turant — toh R² kaise calculate karte ho?

**Answer:**
Is project mein **simulated batches** hain — har batch CSV mein `price` column bhi hai (ground truth already present).

Real production mein yeh problem hoti hai — diamond sell hua, actual price baad mein pata chala.

**Follow-up: Labels delay ke saath aate hain?**
Is project mein assume kiya hai ki labels saath mein aate hain (simplified scenario).

**Follow-up: Labels kabhi na aayein toh?**
Real deployment mein proxy metrics use karo:
- **Prediction distribution shift** — kya model pehle se zyada extreme values predict kar raha hai?
- **Confidence scores** (if probabilistic model) — kya model uncertain ho raha hai?
- **Business metrics** — actual sale price vs predicted price ka comparison (delayed)

---

### Q20: Retraining mein kitna time lagta hai?

**Answer:**
Is project mein data ~54,000 rows hai. Local machine par:
- Data transformation: ~1-2 seconds
- 5 models train: ~10-20 seconds
- Total: **~20-30 seconds**

**Follow-up: Retraining ke dauraan incoming requests ka kya?**
**Not implemented** — incoming predictions still use old model (via registry) while retraining happens.

**Follow-up: Queue hai?**
Not in this project. Ideally: async retraining (background thread/process), predictions continue on old model until new one is ready.

---

# LEVEL 3 — Advanced Technical Questions

---

### Q21: KS test continuous distributions ke liye hai — categorical features ka kya?

**Answer:**
Is project mein KS test sirf **numerical features** par run karta hai: `carat, depth, table, x, y, z`.

Categorical features (`cut, color, clarity`) ke liye KS test **use nahi kiya**.

**Follow-up: Categorical drift kaise handle karein?**
**Not implemented in project.** Ideally:
- **Chi-square test**: Compare frequency distributions of categories
- **Mode shift detection**: Kya most common category badal gayi?
- **Proportion change**: `"Ideal"` cut 60% → 30% ho gayi?

---

### Q22: PSI mein bins consistent kaise rakhe reference aur production ke beech?

**Answer:**
Reference data se bin edges banate hain aur **same edges production par apply karte hain**.

```python
bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)
# Yeh edges reference se calculate hua
expected_counts, _ = np.histogram(expected, bins=bin_edges)
actual_counts, _ = np.histogram(actual, bins=bin_edges)  # same edges!
```

**Follow-up: Production mein reference range se bahar values aayein toh?**
`np.histogram` out-of-range values ko **ignore** karta hai (boundary bins mein nahi aata). Yeh ek limitation hai — extreme outliers miss ho sakte hain.

**Follow-up: Kitne bins? Kyun?**
`bins=10` use kiya — standard choice. Zyada bins = sparse data in each bin = noisy PSI. Kam bins = too coarse-grained = missing subtle shifts.

---

### Q23: Histogram bin errors ki root cause explain karo.

**Answer:**
Jab PSI calculate karte hain toh `np.histogram` ke liye bin edges dete hain. Problem tab aata hai jab **reference data ka min == max** (sabhi values same hain) — `linspace(5, 5, 11)` = `[5,5,5,5...]` = non-monotonic bins.

**Follow-up: Bins monotonically increase kyun nahi karte?**
`linspace(a, b, n)` tabhi monotone hota hai jab `a < b`. Agar `a == b` toh sab same values → numpy error: "bins must increase monotonically".

**Follow-up: np.linspace fix karta hai kaise?**
Linspace evenly spaced values generate karta hai — lekin fix yeh hai ki pehle check karo `if expected.min() == expected.max(): return 0.0` — agar sab same hai toh drift nahi (koi variation hi nahi).

---

### Q24: KS (p-value) + PSI (magnitude) ko single drift decision mein kaise combine karte ho?

**Answer:**
Project mein **OR logic** use ki — dono separate thresholds, agar koi ek fires toh drift:

```python
drift = bool(p_value < 0.05 or psi_score > 0.2)
```

**Follow-up: Weighted score ya separate thresholds?**
Is project mein: **separate thresholds with OR**. No weighting.

**Follow-up: KS drift bolata hai lekin PSI low — kya karein?**
In this project: drift flagged hoga (OR logic). Practically: PSI low matlab magnitude chhoti hai, toh shayad retrain nahi karna chahiye. Future improvement: **AND logic** use karo for fewer false positives.

---

### Q25: Important features mein drift nahi, unimportant mein hai — toh bhi retrain?

**Answer:**
Is project mein haan — koi bhi feature mein drift = system-level drift flag. Feature importance ka koi role nahi.

**Follow-up: Feature importance se weight karo?**
**Not implemented.** Ideal approach:
```python
# Feature importance from trained model
importances = model.feature_importances_  
weighted_drift = sum(drift[f] * importances[f] for f in features)
if weighted_drift > threshold: retrain
```

**Follow-up: Modify kaise karein?**
`DriftDetector.detect_drift()` mein feature-wise drift score ko importance se multiply karo, aggregate weighted score calculate karo, threshold se compare karo.

---

# Performance Monitoring Without Labels

---

### Q26: R² production mein — true values immediately kaise milte hain?

**Answer:**
Short answer: **Is project mein yeh simulated hai.** Production batches mein `price` column already present hai.

Real world mein yeh "labels with delay" problem hai: diamond becha, actual price 2 dinn baad pata chala → delayed evaluation.

**Follow-up: Sirf batch labels simulate kar rahe ho?**
Haan, exactly. Yeh a simplification for demonstration.

**Follow-up: Real deployment mein kya use karein?**
- **Prediction drift** — kya predictions ki distribution badal gayi?
- **Business KPIs** — sales conversions, customer complaints
- **Uncertainty estimation** — Bayesian models mein confidence intervals

---

### Q27: Ground truth delay hone par kaunse proxy metrics use karein?

**Key Points:**
- **Prediction distribution shift**: Pehle model mostly ₹5000-₹8000 predict karta tha, ab ₹2000-₹15000 wide range de raha hai → something wrong
- **Feature drift itself** (already doing this) — input data ka drift output quality ka proxy
- **Prediction confidence** (probabilistic models) — uncertainty badh gayi = model unsure
- **Business outcomes** (delayed) — actual sale price vs prediction

**Follow-up: Proxies validate kaise karein?**
Historical data par test karo: kya yeh proxies tab high the jab actual R² drop hua tha?

---

### Q28: Performance degradation vs temporary anomaly — alag kaise karein?

**Answer:**
Is project mein: **koi distinction nahi**. Single batch ka score dekha jaata hai.

**Follow-up: Smoothing / windowing?**
**Not implemented.** Best practice:
- Last 5 batches ka **moving average** lao
- Agar average < threshold toh alert
- Yeh single-batch anomalies (outlier batch) se bachata hai

**Follow-up: False positive rate?**
From monitoring_log.json: 50 batches mein drift tha 9 mein (~18%), lekin retraining sirf 2 baar hua (v30, v31) — means threshold (score < 0.80) ne false positives filter kiye. Score kabhi 0.80 se neeche nahi gaya in the main run.

---

# System Design & Scalability

---

### Q29: Distributed system mein latest production batch kaun decide karta hai?

**Answer:**
Is project mein simple local `os.listdir()` hai — single machine, no distribution.

**Follow-up: Multiple batches simultaneously aayein?**
Currently no locking mechanism. Race condition possible hogi agar 2 processes simultaneously same folder read karein.

**Follow-up: Locking / coordination?**
**Not implemented.** Real system mein:
- **Message Queue** (Kafka, RabbitMQ) — batches queue mein aate hain, ek ek process hoti hai
- **File locking** (fcntl) — simple local fix
- **Database-based coordination** (Redis lock)

---

### Q30: Batch 5 mein drift detect hua, batch 6 aai retraining ke dauraan — kya hoga?

**Answer:**
Is project mein: batch 6 **queue mein rahegi** until batch 5 ka processing complete ho, kyunki loop sequential hai (ek ek batch process hoti hai).

**Follow-up: Queue karte ho ya parallel?**
Currently: **sequential processing**, no parallelism. Batches ek ek process hoti hain in sorted order.

**Follow-up: Consistency?**
`batch_log.json` ensure karta hai ki ek batch dobara process na ho. Consistency maintain rehti hai.

---

### Q31: Local files → millions of predictions/day tak scale kaise karein?

**Not implemented, but ideally:**

| Component | Local (Current) | Scaled Version |
|---|---|---|
| Data storage | CSV files | S3 / BigQuery |
| Model storage | Local pkl | S3 + model server |
| Registry | JSON file | PostgreSQL / DynamoDB |
| Monitoring log | JSON file | ClickHouse / Elasticsearch |
| Drift detection | In-memory pandas | Distributed Spark job |
| Batch trigger | os.listdir() | Kafka event trigger |

---

### Q32: Dashboard buttons manual — production mein automate kaise karein?

**Answer:**
Is project mein Streamlit buttons se manually trigger karte hain.

**Follow-up: Cron, events, ya streaming?**
- **Cron jobs** (simple): Har 6 ghante monitoring pipeline run karo
- **Event triggers** (smart): Naya batch aaya → trigger monitoring automatically
- **Streaming** (real-time): Kafka par data stream → real-time drift detection

**Follow-up: Orchestration tool?**
- **Apache Airflow** — complex pipelines, DAGs, scheduling
- **Prefect** — simpler, Python-native
- **AWS Step Functions** — agar cloud par

---

### Q33: Multiple models production mein simultaneously — kaise handle karein?

**Not implemented. Ideally:**
- Har model ka apna `model_id` hoga
- Registry mein model_id ke basis par track ho
- Drift detection generic bana do — koi bhi model pass karo, same logic

**Follow-up: Har model ka alag pipeline?**
Ideally: **shared monitoring framework**, model-specific config (thresholds, features, metric).

---

# Edge Cases & Failure Handling

---

### Q34: Retraining midway fail ho gayi toh?

**Answer:**
- Naya model save nahi hoga (pkl write nahi complete hua)
- Registry update nahi hoga (registration try-catch mein hai)
- Purana model production mein **safe rehta hai**

**Follow-up: Checkpointing?**
**Not implemented.** Training pipeline ek shot mein run hoti hai — no intermediate checkpoints.

**Follow-up: Bad model deploy hone se bachao?**
`model_trainer.py` mein rollback logic hai: `if new_score <= old_score: deploy_new_model = False` — yeh protection already hai!

---

### Q35: Production data corrupt ho gayi — system ka response?

**Answer:**
`DataValidation` sabse pehle check karta hai:
- Schema valid hai? (sab 10 columns present?)
- Missing values hain?

Agar validation fail → batch **skip** hota hai, log mein warning jaati hai.

**Follow-up: Kahan validation hoti hai?**
`monitoring_pipeline.py` → Step 1 hi `DataValidation` hai.

**Follow-up: Alert ya reject?**
Currently: **silently skip** with logging. Koi human alert nahi. Ideally: email/Slack notification on validation failure.

---

### Q36: Reference data months baad outdated ho jaaye toh?

**Answer:**
Is project mein reference data **fixed** hai — `artifacts/data/reference_data.csv` exactly wahi hai jo training time par tha.

**Follow-up: Sliding window reference?**
**Not implemented.** Ideal approach:
- Har N batches ke baad reference data ko update karo (last M months ka data)
- Purana reference archive karo

**Follow-up: Trade-off?**
- **Stable reference**: Consistent baseline, but may become outdated
- **Sliding window**: Adapts to slow drift, but may miss gradual concept changes

---

### Q37: "Model Registry Synchronization" issues kya hue?

**Answer:**
Early version mein ek bug tha: `get_next_version()` `len(registry['versions']) + 1` calculate karta tha. Lekin agar `"versions"` key exist nahi karta tha naye JSON mein toh `KeyError` aata tha.

**Key Points:**
- Fix: Registry initialization mein `{"versions": [], "latest_model": None}` empty structure create karo

**Follow-up: Atomic operations?**
JSON file ka write: pehle full dict memory mein banao, phir ek `json.dump()` mein write karo. Agar write beech mein fail ho — file corrupt ho sakti hai. **Ideally**: write to temp file, then atomic rename.

---

### Q38: Har batch mein drift detect ho toh? Infinite retraining loop?

**Answer:**
Is project mein: retraining trigger hoti hai sirf tab jab `drift=True AND score < 0.80`. From actual logs — R² score kabhi 0.80 se below nahi gaya. So 9 batches mein drift tha, lekin retraining sirf 2 baar hua.

**Follow-up: Cooldown period?**
**Not implemented.** Ideally: "last retrain se 24 ghante pehle retrain mat karo" cooldown logic.

**Follow-up: Infinite loop prevent kaise?**
- Cooldown timer
- Max retrains per day limit
- Monitoring the monitor (meta-monitoring)

---

# LEVEL 4 — Expert / Stress Test Questions

---

### Q39: Batch level drift kyun? Per-prediction real-time kyun nahi?

**Answer:**
Statistical tests ke liye **data ki zaroorat hoti hai** — ek prediction par KS test run nahi kar sakte. Minimum ~100-500 samples chahiye reliable result ke liye.

**Follow-up: Real-time drift detection ke liye kya change karein?**
- **ADWIN** (Adaptive Windowing) — streaming algorithm, per-point update
- **Page-Hinkley test** — sequential drift detection
- **CUSUM** — cumulative sum test

**Different architecture:**
- Kafka stream → per-event feature store → rolling window buffer → drift check every N events

---

### Q40: Full retraining kyun? Incremental learning kyun nahi?

**Answer:**
Full retraining simple, predictable, aur stable hai. Incremental learning mein "catastrophic forgetting" ka risk hota hai — naye data par model update karo aur woh purani cheezein bhool jaata hai.

**Follow-up: Incremental learning support karne wale models?**
- `SGDRegressor` (scikit-learn) — `partial_fit()` method hai
- Neural networks — fine-tuning possible
- `PassiveAggressiveRegressor` — online learning

**Follow-up: Full retraining vs incremental trade-offs?**
| | Full Retraining | Incremental |
|---|---|---|
| Stability | ✅ Stable | ❌ Risk of forgetting |
| Speed | ❌ Slow | ✅ Fast |
| Memory | ❌ All data needed | ✅ Stream friendly |
| Complexity | ✅ Simple | ❌ Complex |

---

### Q41: Custom drift detection vs Evidently/NannyML — defend karo.

**Answer:**
Maine khud banake yeh gained kiya:
- **Deep understanding** of KS test mathematics
- **Control** — custom thresholds, custom features, custom aggregation
- **No dependency** on external library versions/APIs
- **Interview mein zyada value** — samajh ke likha hai, copy nahi kiya

**Follow-up: Library better hota kab?**
- Team project mein (speed matters)
- Multiple model types monitor karna ho
- Advanced features: concept drift, data quality reports, HTML dashboards

---

### Q42: Input features par drift kyun? Predictions ya residuals par kyun nahi?

**Answer:**
Input features par drift detect karna **cause** pakadna hai — prediction distribution shift **effect** hai. Cause pehle aata hai!

**Follow-up: Sirf prediction distribution monitor karein?**
Possible, lekin alag features mein opposite drifts cancel out kar sakte hain — net prediction distribution same lag sakti hai lekin andar sab alag ho.

**Follow-up: Kya miss hoga?**
- Kaunsa specific feature responsible hai drift ke liye — pata nahi chalega
- Root cause analysis impossible

---

### Q43: Drift represent karta hai legitimate market shift — adapt karein ya stable rahein?

**Answer:**
Yeh ek philosophical question hai! Agar market genuinely shift hua hai — **adapt karna chahiye**. Agar yeh noise ya temporary anomaly hai — **stable rehna chahiye**.

**Follow-up: Decide kaise karein?**
- Drift ki duration dekho: agar 3+ consecutive batches mein drift hai → likely real shift
- Domain expert se confirm karo
- A/B test: naya model vs purana — business metric par compare karo

---

# Hypothetical Scenarios

---

### Q44: Diamond model deployed hai, globally gold prices spike — kya hoga?

**Answer:**
Gold price is not in the dataset! Diamond price se gold ka direct correlation nahi hai in our features. But jo real impact ho sakta hai: market mein diamonds ki demand shift ho (correlated economic event).

**Follow-up: Drift detect hoga?**
Agar diamds ki demand badli aur actual sales data (carat/price distribution) shift hua → **haan, KS/PSI detect karega**.

**Follow-up: Immediately retrain karein ya wait?**
Wait — 1-2 batches ka data collect karo. Ek spike temporary ho sakta hai. Agar 3 consecutive batches mein drift confirm ho toh retrain.

---

### Q45: Fraud detection model ke liye drift detection alag kaise hogi?

**Answer:**
Fraud detection → **classification** problem. Sab kuch change ho jaata:

| Aspect | Diamond (Regression) | Fraud (Classification) |
|---|---|---|
| Performance metric | R² | F1, AUC-ROC, Precision-Recall |
| Drift on target | Price distribution | Fraud rate (% of fraud cases) |
| Critical metric | Raw accuracy | False Negative Rate (missed frauds) |

**Follow-up: Class imbalance ka effect?**
Fraud data mein 99% normal, 1% fraud. PSI calculate karte time stratified sampling zaroor karna — nahi toh 1% minority class completely miss ho jaata hai drift calculation mein.

---

### Q46: "Model 5 times retrained this month — too much" — stakeholder ko kya bolte ho?

**Answer:**
Show them data:
- "Har baar retrain hone ke baad R² score improve hua — dekho monitoring_log.json"
- "Agar retrain nahi karte toh R² X.XX se girkar Y.YY aa jaata — estimated revenue loss Z"
- Retraining cost vs prediction error cost compare karo

**Follow-up: False positives reduce karne ke liye?**
- Threshold tighten karo: `score < 0.75` instead of `< 0.80`
- AND logic: KS AND PSI dono drift bolein tabhi flag
- Rolling average: 3 consecutive batches mein drift hone par trigger

---

### Q47: Drift detect hua, retrain kiya, performance worse — kya galat hua?

**Answer:**
Possible reasons:
1. **Training data ka drift** — reference data bhi purana ho gaya, naye data par train kiya lekin woh bhi representative nahi
2. **Overfitting** — chhote batch par retrain kiya
3. **Feature engineering mismatch** — transform kiya differently
4. **Retraining data quality** — production batch corrupted tha

**Follow-up: Debugging steps?**
1. Check karo: naye model ka training R² vs test R² — overfitting?
2. Validation data distribution check karo
3. Production batch ki distribution manually inspect karo
4. Reference data ko update karne ki zaroorat hai?

**Follow-up: Worse model se bachao?**
Already implemented: `model_trainer.py` mein `if new_score <= old_score: deploy_new_model = False` → rollback automatic!

---

### Q48: AWS Lambda par deploy karein strict cost constraints ke saath — kya change karein?

**Answer:**
Lambda mein: stateless, 15 min max execution, 10 GB RAM max, ~512 MB disk.

**Changes needed:**
- Drift detection ek Lambda function (event-triggered)
- Model inference alag Lambda (lightweight)
- Full retraining → **nahi chalega** on Lambda (too slow, too much memory)→ move to SageMaker Training Job
- Model storage → S3
- Registry → DynamoDB instead of JSON file

**Follow-up: Serverless drift detection?**
Haan possible — drift detection ~5-10 seconds hai, Lambda mein fit:
- Trigger: S3 event (new batch file uploaded)
- Lambda: download batch, run KS + PSI, write result to DynamoDB
- If drift: trigger Step Function for retraining

---

# "Why Not X?" Questions

---

### Q49: SPC (Statistical Process Control) charts kyun nahi use kiye?

**Answer:**
SPC charts (X-bar charts, control charts) sequential data ke liye designed hain — time series par outliers detect karte hain single metric mein. KS test **distribution comparison** karta hai jo zyada comprehensive hai.

**Follow-up: Difference?**
- SPC: "Kya yeh value control limits ke bahar hai?" (point anomaly)
- KS: "Kya distributions same hain?" (overall shift)

**Follow-up: Combine kar sakte hain?**
Haan! SPC for per-batch mean monitoring + KS for distribution comparison = multi-layered detection.

---

### Q50: Shadow model kyun nahi use kiya continuously validate karne ke liye?

**Answer:**
Shadow model: production model ke parallel ek aur model chalta hai, dono predictions compare karte hain.

**Not implemented.** Reasons:
- Double compute cost
- Complexity zyada
- For this project scope, drift+performance was sufficient

**Follow-up: Shadow model vs drift detection?**
Shadow model tells you **if models disagree** — drift detection tells you **why** (which features changed).

---

### Q51: Adversarial validation for drift kyun nahi?

**Answer:**
Adversarial validation: reference + production data mix karo, label karo (0=reference, 1=production), ek classifier train karo. Agar classifier high AUC deta hai → data alag hai (drift!).

**Not implemented.** Interesting approach lekin:
- Har batch mein classifier train karna expensive
- KS + PSI simpler aur equally effective for numerical data

**Follow-up: Kab better?**
- Complex high-dimensional data (images, text) mein
- Jab features correlated hain (univariate tests miss karte hain)

---

### Q52: Time-based schedule triggers kyun nahi?

**Answer:**
Schedule mein: chahe zaroorat ho ya na ho, retrain hoga. Scientific evidence-based triggering smarter hai.

**Follow-up: Slow, continuous drift miss hoga?**
Haan! KS test batches ke beech compare karta hai — agar har batch mein thoda thoda drift hai toh individual test miss karega. **Cumulative drift detection** needed hoga.

**Follow-up: Gradual degradation miss hoga?**
Possible. Solution: rolling baseline — last 10 batches ko reference maano, current batch se compare karo.

---

### Q53: Multivariate drift detection kyun nahi?

**Answer:**
Is project mein: ek ek feature separately test kiya (univariate). Multivariate: saare features ek saath consider karo.

**Follow-up: Univariate mein kya miss hota hai?**
Feature correlations! Agar `x` aur `y` dono thode change karein lekin ek direction mein — individually undetectable, jointly significant.

**Follow-up: Multivariate implement kaise?**
- **Maximum Mean Discrepancy (MMD)** — kernel-based test on full feature space
- **PCA + KS** — principal components par test
- Scikit-learn mein `KernelDensity` use karke

---

# Production & Deployment

---

### Q54: Automated retraining mein model explainability kaise karein?

**Not implemented. Ideally:**
- **SHAP values** store karo for each model version
- "Model v31 zyada weight deta hai `carat` ko because training data mein carat-price correlation shift hua"
- Har model version ke saath SHAP summary plot save karo

**Follow-up: Regulatory requirements?**
- Model card banao: training data, performance metrics, known limitations
- Version-specific documentation
- Human review before deploy (human-in-the-loop)

---

### Q55: Streamlit ko enterprise ke liye productionize kaise karein?

**Not implemented. Ideally:**
- **Authentication**: OAuth2 / SSO (not just open URL)
- **Authorization**: Admin vs Viewer roles
- **Deployment**: Docker + Kubernetes (not just `streamlit run`)
- **Better UI framework**: React + FastAPI backend instead of Streamlit
- **Audit logs**: Kaun kab run training pipeline kiya

---

### Q56: CI/CD pipelines ke saath integration kaise?

**Not implemented. Ideally:**

```
Code push → GitHub Actions →
  1. Unit tests run (pytest)
  2. Model validation tests
  3. Docker image build
  4. Deploy to staging
  5. Integration tests
  6. Deploy to production (if all pass)
```

**Retrained model ke tests:**
- Score > baseline threshold
- Feature distributions within expected ranges
- Inference latency < SLA

---

# Failure & Redesign

---

### Q57: Agar aaj se dobara banate toh kya alag karte?

**Key Points:**
- **Better**: Sliding window reference data — fixed reference outdated ho jaata hai
- **Better**: Categorical drift detection bhi add karta (`cut`, `color`, `clarity`)
- **Better**: Async retraining — predictions block nahi honi chahiye during retraining
- **Mistake**: `data_ingestion.py` ek dead file ban gayi — training pipeline mein integrate karni chahiye thi
- **Worked well**: OR logic for drift, JSON registry, batch_log dedup — yeh sab clean solutions the

---

### Q58: Monitoring pipeline memory crash — debug kaise karein?

**Potential causes:**
- 50 CSVs simultaneously load ho gayein memory mein
- Reference data (7.4 MB) + batch + model (14 MB) = 40+ MB per iteration × batches

**Follow-up: Profiling tools?**
```python
import tracemalloc
tracemalloc.start()
# … code …
snapshot = tracemalloc.take_snapshot()
```
Ya: `memory_profiler` library, `line_profiler`

**Follow-up: Memory optimization?**
- Batch ko chunks mein process karo (`pd.read_csv(chunksize=1000)`)
- Model: load once, reuse — load inside loop mat karo
- Explicit `del df; gc.collect()` after each batch

---

### Q59: Drift detection latency 30 sec → 1 sec kaise karein?

**Bottleneck:**
- CSV load: ~1-2 sec (7.4 MB reference)
- KS test + PSI per feature: ~0.5 sec × 6 = 3 sec
- Total: reasonable for batch, not for real-time

**Follow-up: Optimization:**
- Reference data ko **RAM mein pre-load** karo (load once, not every batch)
- **Vectorized PSI** instead of Python loop
- **Sampling**: 1000 rows se test karo, poora data nahi
- **Approximate KS** using pre-computed CDFs

**What to sacrifice:**
- Accuracy thodi kam — sample instead of full data
- Categorical drift skip karo (already not doing it)

---

### Q60: Too many false positive drift alerts — reduce kaise karein?

**Analysis:**
Look at: kaunse features mein drift tha? Kaunsa threshold fire kar raha hai (KS ya PSI)?

**Solutions:**
- **Raise p-value threshold**: 0.05 → 0.01 (stricter)
- **Raise PSI threshold**: 0.2 → 0.25
- **Switch OR → AND**: KS AND PSI dono → fewer alerts
- **Rolling window**: 3 batches mein drift hone par hi flag
- **Feature weighting**: Unimportant features ki drift ignore karo

**Validate changes:**
Historical data par test karo — kya true drifts (jo actual retraining needed tha) still detected hote hain?

---

### Q61: Naya developer add karna hai new drift algorithm — kitna extensible hai?

**Current state:**
`DriftDetector` class mein `detect_drift()` method hai. New algorithm add karna = this method modify karna required.

**Follow-up: Kya change karna padega?**
`drift_detector.py` mein method add karo + `detect_drift()` mein call karo. High coupling.

**Follow-up: Pluggable design kaise banta?**
```python
class DriftDetector:
    def __init__(self, detectors=None):
        self.detectors = detectors or [KSDetector(), PSIDetector()]
    
    def detect_drift(self, ref, cur):
        results = {}
        for detector in self.detectors:
            results.update(detector.run(ref, cur))
        return results
```
Plugin-style architecture — new detector add karo, class touch mat karo.

---

# Meta-Level & System Thinking

---

### Q62: Is self-healing system ka ROI kaise measure karein?

**Metrics to show:**
- **Model uptime at target performance**: e.g., 95% of time R² > 0.90
- **Time to detect drift**: how quickly system caught it vs manual monitoring
- **Engineer hours saved**: no manual monitoring needed
- **Prevented outages**: how many times auto-retrain prevented major accuracy drop

**Follow-up: Prevented failures quantify kaise karein?**
Counterfactual analysis: "Agar retrain nahi kiya hota toh R² X ho jaata. X score par estimated revenue error Y rupees tha. Retrain kiya toh Z rupees bachaye."

---

### Q63: Monitoring system ko monitor kaun karta hai?

**Answer:**
Is project mein: **koi meta-monitoring nahi hai**. Yeh ironic hai!

**Ideally:**
- Drift detection ka unit test: synthetic drifted data dalo, check karo ki flag hua
- `batch_log.json` **age** monitor karo — agar 24 ghante se koi batch process nahi hua → alert
- Monitoring pipeline ki last success timestamp track karo
- Health check endpoint: `GET /health` → returns pipeline status

**Follow-up: Unit tests for statistical tests?**
Not implemented. Ideally:
```python
def test_ks_detects_drift():
    ref = pd.Series(np.random.normal(0, 1, 1000))
    drifted = pd.Series(np.random.normal(5, 1, 1000))  # Clearly different
    _, p_value = ks_2samp(ref, drifted)
    assert p_value < 0.05  # Should detect drift
```

---

### Q64: Non-technical executive ko explain karo — automatic retraining quarterly manual se better kyun?

**Simple explanation:**
"Imagine aapka salesperson ko last year ki price list ke basis par customers ko quote de raha hai. Market badal gayi — competitor prices drop, new regulations — lekin salesperson ki list purani hai. Hum quarterly update karte the — 3 months purani list!

Mera system: jaise hi market data changes → automatically salesperson ki list update ho jaati hai. Koi manual kaam nahi, koi delay nahi, koi missed opportunity nahi."

**Business case:**
- Accuracy drop = wrong predictions = lost deals / wrong pricing
- Each 1% R² drop = X rupees revenue impact
- System prevents this automatically, 24/7, without hiring a monitoring engineer

**Risks of automation:**
- Bad model auto-deploy ho sakta hai (mitigated by rollback logic)
- Unexpected market shifts ko model wrongly adapt kar sakta hai
- Solution: human approval required for models affecting >10% revenue

---

### Q65: Ek question jo mujhe umeed hai nahi puchha jayega — aur ab answer karo.

**The uncomfortable question:**
*"Tumhare 50 production batches simulated hain — real production data nahi hai. Yeh system actually kabhi tested nahi hua real drift ke saath?"*

**Honest answer:**
Haan, yeh fair criticism hai. Batches `test_drift_data_maker.py` ne banaye — manually noise add karke drift simulate kiya. Real production data bohot zyada unpredictable hota hai:
- Sudden data format changes
- Sensor failures causing missing values
- Concept drift (relationship between features and target changes)

Is project ne **architecture aur logic** sahi kar diya hai. Real production deployment ke liye:
- Actual data pipeline (database/API se)
- Chaos testing (inject failures deliberately)
- Shadow deployment (run alongside existing system)
...ki zaroorat hogi.

---

# Bonus: Curveball Questions

---

### Q66: Streaming data dene par kya components break karein?

**What breaks:**
1. **`os.listdir()` in monitoring pipeline** — file system batches assume karta hai, streams nahi
2. **KS test and PSI** — minimum sample size chahiye, single events par nahi chalta
3. **`pd.read_csv()`** — file read karta hai, stream consume nahi karta
4. **Batch dedup logic** — stream mein "batch" concept nahi hota

**What works:**
- `DataValidation` logic — row-level checks reusable hain
- Model inference — single row predict kar sakta hai
- Registry — model lookup kaam karega

---

### Q67: Covariate shift vs Concept drift — tu actually kaunsa detect karta hai?

**Definitions:**
- **Covariate shift**: Input feature X ki distribution badal gayi (P(X) changes), relationship P(Y|X) same hai
- **Concept drift**: Relationship itself badal gayi — same X deta hai alag Y (P(Y|X) changes)

**Mera system actually detects:**
**Covariate shift!** — Main sirf input features (`carat`, `depth`, etc.) ki distribution compare karta hoon. Main `price` (Y) ki distribution ya X→Y relationship monitor nahi karta.

**Concept drift kaise detect karein?**
- Monitor residuals (predicted - actual) over time
- Agar residuals ka mean shift ho raha hai → concept drift
- Monitor R² trend (already partially doing this)

---

### Q68: KS test drift detection ke liye mathematically valid kyun hai?

**Simple math explanation:**
KS statistic: `D = max|F_ref(x) - F_prod(x)|`  
Jahan F = CDF (cumulative distribution function)

Under H₀ (same distribution):  
`D_n ≈ 0` (CDFs almost same honge)

Under drift (different distributions):  
`D_n` large hoga

KS theorem prove karta hai ki if H₀ true hai, D follows the Kolmogorov distribution. P-value = "Probability ki itna bada D randomly observed ho agar distributions same hoon." 

p < 0.05 matlab: 5% se kam chance ki yeh randomly hua → reject H₀ → **drift detected**.

---

### Q69: Adversary drift detection game karne ki koshish kare — kya hoga?

**Scenario:**
Adversary production batches mein carefully crafted data daale jo KS/PSI thresholds ke just under rahein — drift trigger na kare — lekin model ko fool kare.

**Impact:**
- Drift detection miss ho jaayega (adversary wins)
- Model wrong predictions karega without system knowing
- Performance monitor agar R² check karega — woh bhi adversarially crafted ho sakta hai agar adversary has price column access

**Mitigation (not implemented):**
- Multiple independent drift detectors
- Anomaly detection on prediction distributions
- Rate limiting on batch processing
- Auth on who can upload production batches

---

### Q70: Multi-output model (price + quality + cut simultaneously predict kare)?

**Answer:**
Current system: single output (price). Multi-output would need:
- Multiple performance metrics (one per output)
- Drift detection per output's residuals
- Retraining decision: koi bhi ek output drift kare toh retrain?

**Changes:**
- `performance_monitor.py`: `R² per output` → aggregate (mean or worst)
- `model_trainer.py`: `MultiOutputRegressor` wrapper
- `prediction_pipeline.py`: Multiple output columns instead of single value

---

### Q71: Python suddenly 10x slow — kya pehle break hoga?

**Order of impact (slowest first):**

1. **KS test** — `scipy.stats.ks_2samp` internally sorts data — O(n log n) → 10x slower = noticeable
2. **PSI histogram computation** — numpy operations, will feel it
3. **Model inference** (sklearn predict) — numpy backed, moderate impact
4. **CSV loading** (pandas) — I/O bound mostly, less impacted

Pure Python parts mein (loops in `calculate_psi`) would hurt most. Fix: switch to fully vectorized numpy/numba.

---

### Q72: Model retrain hua but production performance improve nahi hua — debug karo.

**Step-by-step debugging:**

```
1. Check: Kya new model register hua?
   → model_registry.json: naya version latest_model hai?

2. Check: Naye model ka training/test R²?
   → Kya yeh simulation tha? Training R² achha tha?

3. Check: Kya preprocessing sahi hua?
   → Preprocessor.pkl refit hua ya purana use hua?

4. Check: Production batch aur training data ka distribution?
   → Manually plot karo — kya training data drift resolve karta hai?

5. Check: Prediction pipeline sahi model load kar rahi hai?
   → Add logging: print karo model_path jo load ho raha hai

6. Check: Reference data outdated toh nahi?
   → Naya model purane reference par train hua, production kuch aur dikha raha hai
```

Most likely root cause: **reference data outdated** — model retrained on old reference that doesn't match current production distribution anymore. Fix: update reference data before retraining.
