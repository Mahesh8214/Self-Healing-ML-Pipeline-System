# Self-Healing ML Pipeline --- Antigravity Implementation Prompt

## ROLE

You are a senior MLOps engineer, ML platform architect, Python backend
engineer, and Streamlit developer.

You are working on my existing project:

**"Self-Healing ML Pipeline for Data Drift Detection &
Auto-Retraining"**

This is an **EXISTING WORKING CODEBASE**.

Do **NOT** rebuild the project from scratch.\
Do **NOT** replace working functionality unnecessarily.\
Do **NOT** create duplicate implementations if equivalent functionality
already exists.

Your job is to first understand the existing architecture and then
incrementally improve it.

------------------------------------------------------------------------

## PROJECT OBJECTIVE

The system should demonstrate a realistic self-healing ML lifecycle:

``` text
Production Model
      ↓
Production / Incoming Data
      ↓
Data Monitoring
      ↓
Drift Detection
      ↓
Feature-Level Drift Diagnosis
      ↓
Model Performance Assessment
      ↓
Retraining Trigger
      ↓
Background Retraining
      ↓
Challenger Model Evaluation
      ↓
Champion vs Challenger Comparison
      ↓
Validation / Quality Gate
      ↓
Promote or Reject Challenger
      ↓
Updated Production Model
      ↓
Monitoring Continues
```

The dashboard should make this entire lifecycle visible and
understandable.

The project must demonstrate a **REAL MLOps use case**, not just provide
buttons that synchronously execute Python functions.

------------------------------------------------------------------------

## PHASE 0 --- ANALYZE THE EXISTING CODEBASE FIRST

Before modifying **ANY** code:

1.  Inspect the complete project structure.

2.  Identify:

    -   Streamlit entry point
    -   All Streamlit pages
    -   Current dashboard/home page
    -   `drift_dashboard.py`
    -   Training pipeline code
    -   Drift detection implementation
    -   Model training implementation
    -   Model evaluation implementation
    -   Model registry/versioning logic
    -   Deployment/promotion logic
    -   Database/storage implementation
    -   MLflow usage, if any
    -   FastAPI usage, if any
    -   Existing background job implementation, if any
    -   Session state usage
    -   Current navigation implementation

3.  Trace the existing workflow for:

    -   "Run Monitor"
    -   "Run Training Pipeline"
    -   Drift detection
    -   Retraining
    -   Model promotion

4.  Identify which functionality already exists and **REUSE it**.

5.  Before implementation, create a short implementation plan
    containing:

    -   Files that need modification
    -   Files that need creation
    -   Existing functions/components that will be reused
    -   Architecture changes required

Then implement the changes.

------------------------------------------------------------------------

## PART 1 --- FIX "RUN MONITOR" NAVIGATION

When the user clicks:

**"Run Monitor"**

the application should navigate to the existing monitoring/drift page.

Prefer the existing file:

`ui/pages/drift_dashboard.py`

Use Streamlit's supported page navigation mechanism for the project's
current Streamlit version.

For example, if compatible:

``` python
st.switch_page("pages/drift_dashboard.py")
```

Do not break the existing multipage navigation.

The monitor action should trigger or display the latest monitoring run.

------------------------------------------------------------------------

## PART 2 --- REDESIGN DRIFT DASHBOARD

Transform `drift_dashboard.py` into a professional:

**"Drift Monitoring & Self-Healing Center"**

The page should have these sections.

### A. SYSTEM STATUS

At the top show:

System Status: - HEALTHY - DRIFT DETECTED - RETRAINING - RECOVERED -
ACTION REQUIRED

Also show:

-   Current production model version
-   Last monitoring timestamp
-   Dataset / data source
-   Latest monitoring run ID
-   Current pipeline state

Use cards/metrics where appropriate.

Do **NOT** hardcode fake values.

Values must come from actual pipeline outputs, persisted run state, or
clearly labeled demo data if the project has a demo mode.

### B. DRIFT SUMMARY

Show:

-   Total Features
-   Number of Drifted Features
-   Drift Percentage
-   Overall Drift Detected: Yes/No
-   Retraining Required: Yes/No
-   Drift threshold

Example:

``` text
Total Features: 12
Drifted Features: 3
Drift Percentage: 25%
Overall Drift: DETECTED
Retraining Required: YES
```

All values must be dynamically calculated.

### C. FEATURE-LEVEL DRIFT TABLE

Display every monitored feature.

Columns:

-   Feature
-   Drift Metric
-   Drift Score
-   Threshold
-   Status

Example:

``` text
age                  PSI    0.05    0.20    Stable
income               PSI    0.34    0.20    Drifted
transaction_amount   PSI    0.41    0.20    Drifted
```

Visually distinguish:

-   Stable
-   Warning
-   Drifted

Do not only show an overall drift result.

The user must be able to understand **EXACTLY which features caused
drift**.

If the current implementation uses different statistical tests for
numerical/categorical features, preserve those tests and display the
correct metric/test name.

------------------------------------------------------------------------

## PART 3 --- FEATURE DRIFT EXPLORER

Allow the user to select a feature from the drift results.

Preferably prioritize drifted features.

For the selected feature show:

-   Feature Name
-   Drift Score
-   Drift Threshold
-   Drift Status
-   Statistical Test / Metric Used

Then visualize:

**REFERENCE DATA DISTRIBUTION**\
vs\
**CURRENT / PRODUCTION DATA DISTRIBUTION**

For numerical features use an appropriate visualization such as: -
Histogram - KDE/distribution comparison

For categorical features use: - Grouped bar chart - Frequency/proportion
comparison

The graph must use **REAL reference and current data**.

The purpose is to answer:

> "What changed in this feature?"

Do not generate random chart data in production mode.

------------------------------------------------------------------------

## IMPORTANT CONCEPTUAL REQUIREMENT

Do **NOT** claim:

> "Drift disappeared after retraining."

Retraining a model does **NOT** remove data drift.

Data drift describes a change in the input data distribution.

The correct story is:

``` text
Data Distribution Changed
        ↓
Drift Detected
        ↓
Old Model Performance May Degrade
        ↓
Retraining Triggered
        ↓
New Model Learns From Updated Data
        ↓
Model Performance Recovers
```

Keep these concepts separate:

1.  **DATA DRIFT**
    -   Reference data vs Current data
2.  **MODEL PERFORMANCE RECOVERY**
    -   Champion model vs Challenger model

Never show a misleading graph called:

**"Drift Before vs Drift After Retraining"**

unless the system is genuinely comparing two different data windows.

------------------------------------------------------------------------

## PART 4 --- SELF-HEALING PIPELINE STATUS

Add a visual pipeline status component.

Stages:

1.  Data Validation
2.  Drift Detection
3.  Retraining Trigger
4.  Model Training
5.  Model Evaluation
6.  Champion vs Challenger
7.  Validation / Quality Gate
8.  Model Promotion or Rejection

Each stage should have states such as:

-   WAITING
-   RUNNING
-   COMPLETED
-   FAILED
-   SKIPPED

Example:

``` text
✓ Data Validation
✓ Drift Detection
  └── 3 features drifted
✓ Retraining Triggered
⏳ Model Training
○ Model Evaluation
○ Champion vs Challenger
○ Deployment Decision
```

Pipeline state must reflect the **REAL execution state**.

Do not simulate progress percentages that are not measurable.

If exact training percentage is unavailable, show stage-level progress
instead.

------------------------------------------------------------------------

## PART 5 --- FIX LONG-RUNNING TRAINING UX

Currently when the user clicks:

**"Run Training Pipeline"**

the application waits a long time while training runs.

This is poor architecture for a long-running ML task.

Refactor this so the UI does **NOT** remain blocked unnecessarily.

Desired architecture:

``` text
User clicks Run Training
        ↓
Create Training Job
        ↓
Return control to UI
        ↓
Background Worker executes training
        ↓
Persist Job Status
        ↓
UI polls/refreshes Job Status
        ↓
Display current pipeline stage
        ↓
Training completes
        ↓
Display evaluation results
```

First inspect the current deployment architecture.

Implement the simplest reliable background-job architecture compatible
with the existing project.

Possible approaches include:

-   Thread/process-based worker for a simple portfolio/demo deployment
-   Redis + RQ
-   Redis + Celery
-   FastAPI background execution
-   Another appropriate existing mechanism

Do **NOT** introduce Redis/Celery purely for complexity.

Choose based on the current architecture and deployment environment.

### IMPORTANT

The background job must not depend only on `Streamlit session_state` for
persistent execution state.

Persist job metadata where practical.

Suggested job fields:

``` text
job_id
status
current_stage
created_at
started_at
completed_at
error_message
model_version
trigger_reason
```

Possible statuses:

``` text
QUEUED
RUNNING
COMPLETED
FAILED
```

------------------------------------------------------------------------

## PART 6 --- TRAINING JOB UI

When training starts, immediately show something like:

``` text
Training Job #<id>
Status: RUNNING

Current Stage: Model Training
Elapsed Time: ...

Pipeline:

Data Validation       Completed
Drift Detection       Completed
Retraining Trigger    Completed
Model Training        Running
Model Evaluation      Waiting
Quality Gate          Waiting
Model Promotion       Waiting
```

Do not show a fake exact percentage unless the training algorithm
provides meaningful measurable progress.

Stage-level progress is sufficient.

The user should be able to navigate away and return without losing the
job status, if supported by the current architecture.

------------------------------------------------------------------------

## PART 7 --- CHAMPION VS CHALLENGER

After retraining completes, show a professional comparison.

-   **Champion** = current production model
-   **Challenger** = newly trained model

Compare relevant metrics dynamically.

For example:

  Metric        Champion   Challenger   Change
  ----------- ---------- ------------ --------
  Accuracy         82.3%        91.4%    +9.1%
  F1 Score          0.79         0.90    +0.11
  Precision         0.81         0.92    +0.11
  Recall            0.77         0.88    +0.11

Use the metrics appropriate to the existing ML problem.

Do not assume accuracy is always the primary metric.

Use the project's configured primary metric.

Clearly show:

-   Champion Model Version
-   Challenger Model Version

------------------------------------------------------------------------

## PART 8 --- MODEL QUALITY GATE

Implement or preserve a model promotion rule.

Example concept:

``` text
Promote challenger only if:

challenger_primary_metric >
champion_primary_metric + minimum_improvement

AND

all required validation checks pass.
```

The exact rule should integrate with the project's existing validation
logic.

Display:

**Promotion Decision**

-   PROMOTED
-   REJECTED

And explain why.

Examples:

> "Challenger promoted because F1 improved from 0.79 to 0.90 and all
> validation checks passed."

or

> "Challenger rejected because improvement was below the configured 2%
> minimum threshold."

Do not automatically deploy every newly trained model.

------------------------------------------------------------------------

## PART 9 --- MODEL PERFORMANCE RECOVERY

Create a section:

**"Model Performance Recovery"**

Show:

``` text
Old Production Model Performance
        ↓
Drift Event
        ↓
Retraining
        ↓
Challenger Performance
        ↓
Promotion Decision
        ↓
New Production Model
```

Clearly communicate:

Drift remains a property of the changed data.

The **"recovery"** refers to **MODEL PERFORMANCE** adapting to the new
data distribution.

------------------------------------------------------------------------

## PART 10 --- SELF-HEALING EVENT TIMELINE

Create an event timeline for each monitoring/retraining cycle.

Example:

``` text
10:30:02
Monitoring started

10:30:05
Data validation passed

10:30:08
Drift detected

Drifted Features:
- income
- transaction_amount
- location

10:30:10
Retraining triggered

10:32:45
Challenger model trained

10:32:50
Evaluation completed

Champion F1: 0.79
Challenger F1: 0.90

10:32:52
Quality gate passed

10:32:55
Model v4 promoted to production

10:32:56
System status changed to RECOVERED
```

These events must be generated from actual pipeline events/timestamps.

Persist them if the existing architecture has a database or suitable
storage.

------------------------------------------------------------------------

## PART 11 --- AUTOMATIC SELF-HEALING

The architecture should support:

``` text
Monitoring
    ↓
Drift threshold exceeded
    ↓
Automatically trigger retraining
    ↓
Train challenger
    ↓
Evaluate
    ↓
Quality gate
    ↓
Promote only if better
```

However, retain manual control where useful.

Recommended modes:

**AUTO-HEALING: ON/OFF**

When **ON**:

Significant drift automatically triggers retraining.

When **OFF**:

Drift is detected and the dashboard shows:

**"Retraining Recommended"**

with a:

**"Trigger Retraining"** button.

Avoid accidentally triggering multiple retraining jobs for the same
drift event.

Implement protection against duplicate active jobs.

------------------------------------------------------------------------

## PART 12 --- DEMO MODE VS PRODUCTION MODE

If compatible with the current project, introduce two modes.

### DEMO MODE

Purpose:

Recruiter/interviewer demonstration.

Characteristics:

-   Smaller dataset
-   Fast training
-   Reduced model complexity
-   Limited hyperparameter search
-   Target execution time approximately 20--60 seconds where feasible

### PRODUCTION MODE

Characteristics:

-   Full dataset
-   Full validation
-   Complete training pipeline
-   Production-quality configuration
-   Asynchronous execution

Do not fake results in Demo Mode.

Demo Mode must still execute the real pipeline, just using a reduced
workload/configuration.

------------------------------------------------------------------------

## PART 13 --- DASHBOARD NAVIGATION

The desired user journey is:

``` text
HOME DASHBOARD
      ↓
RUN MONITOR
      ↓
DRIFT MONITORING CENTER
      ↓
DRIFT SUMMARY
      ↓
FEATURE-LEVEL DRIFT ANALYSIS
      ↓
SELECT FEATURE
      ↓
REFERENCE VS CURRENT DISTRIBUTION
      ↓
RETRAINING REQUIRED?
      ↓
YES
      ↓
AUTO-TRIGGER OR MANUAL TRIGGER
      ↓
TRAINING JOB STATUS
      ↓
CHAMPION VS CHALLENGER
      ↓
QUALITY GATE
      ↓
PROMOTE / REJECT
      ↓
MODEL PERFORMANCE RECOVERY
      ↓
SELF-HEALING EVENT TIMELINE
```

The UI should make this flow obvious.

------------------------------------------------------------------------

## PART 14 --- ERROR HANDLING

Handle these cases gracefully:

-   No reference dataset
-   No current dataset
-   No production model
-   Drift calculation fails
-   Training fails
-   Evaluation fails
-   Model registry fails
-   Promotion/deployment fails
-   User refreshes page during training
-   Duplicate training request
-   Background job crashes

Never leave the UI stuck indefinitely.

Show useful error messages.

Persist **FAILED** state with an error reason.

------------------------------------------------------------------------

## PART 15 --- CODE QUALITY

Requirements:

-   Keep UI logic separate from ML pipeline logic.
-   Do not put model training code directly inside Streamlit button
    handlers.
-   Reuse existing services/functions.
-   Avoid duplicate drift detection implementations.
-   Avoid duplicate model evaluation logic.
-   Avoid hardcoded paths.
-   Use configuration/environment variables where appropriate.
-   Add type hints where practical.
-   Add logging.
-   Preserve existing working functionality.
-   Keep deployment compatibility.
-   Do not add unnecessary dependencies.
-   Update `requirements.txt` only if genuinely required.

------------------------------------------------------------------------

## PART 16 --- TESTING

After implementation test the complete workflow.

### Test Case 1: No drift detected

Expected: - Monitoring completes. - Features show Stable. - No
retraining triggered.

### Test Case 2: Drift detected

Expected: - Correct features identified. - Feature distribution graph
works. - Retraining recommendation appears.

### Test Case 3: Auto-healing enabled

Expected: - Drift automatically triggers exactly one training job.

### Test Case 4: Challenger better

Expected: - Quality gate passes. - Challenger promoted.

### Test Case 5: Challenger worse

Expected: - Challenger rejected. - Production model remains unchanged.

### Test Case 6: Training failure

Expected: - Job marked FAILED. - Error displayed. - Production model
unaffected.

### Test Case 7: Page refresh during training

Expected: - Job status can be recovered from persisted state where
architecture permits.

------------------------------------------------------------------------

## FINAL DELIVERABLE

After implementation, provide:

1.  Summary of the existing architecture you discovered.
2.  List of files modified.
3.  List of new files created.
4.  Explanation of the final architecture.
5.  Exact workflow:
    `Monitor → Detect → Diagnose → Retrain → Evaluate → Validate → Promote/Reject`
6.  Explain how background training works.
7.  Explain where training job state is stored.
8.  Explain how feature-level drift is calculated and visualized.
9.  Explain the champion/challenger promotion logic.
10. Explain how duplicate retraining jobs are prevented.
11. Give exact commands to run the complete project locally.
12. Mention any new environment variables or dependencies required.
13. Verify that existing functionality was not unnecessarily removed.

------------------------------------------------------------------------

## IMPORTANT FINAL INSTRUCTION

Do not just redesign the UI.

The underlying pipeline state, drift results, training jobs, evaluation
metrics, and model promotion decisions must be connected to **REAL
backend/pipeline outputs**.

Do not hardcode:

-   Fake metrics
-   Fake drift scores
-   Fake progress
-   Fake timestamps
-   Fake model versions

Prioritize a **functional end-to-end MLOps workflow** over visual
complexity.
