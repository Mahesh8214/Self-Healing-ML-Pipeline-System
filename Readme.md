\documentclass[11pt]{article}

\usepackage{geometry}
\geometry{margin=1in}

\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{titlesec}

\titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}

\definecolor{codegray}{gray}{0.95}

\lstset{
backgroundcolor=\color{codegray},
basicstyle=\ttfamily\small,
breaklines=true
}

\title{\textbf{Self-Healing ML Pipeline System}\\
\large Data Drift Detection \& Automatic Model Retraining}

\author{Mahesh Singh}
\date{}

\begin{document}

\maketitle

\section*{Abstract}

Modern Machine Learning systems often fail silently when the underlying data distribution changes. 
This phenomenon, known as \textbf{data drift}, can significantly degrade model performance in production environments.

This project implements a \textbf{Self-Healing Machine Learning Pipeline} capable of detecting distribution changes, 
monitoring model performance, and automatically retraining models when necessary.

The system integrates:

\begin{itemize}
\item Drift detection using statistical tests
\item Performance monitoring
\item Automatic retraining
\item Model versioning
\item Visualization dashboard
\end{itemize}

The objective is to create a \textbf{production-oriented ML system} that maintains model reliability over time.

\section{Problem Statement}

Traditional ML workflows follow a static process:

\begin{lstlisting}
Train -> Deploy -> Forget
\end{lstlisting}

However, in real-world production systems:

\begin{itemize}
\item Data distributions evolve
\item Model performance degrades
\item Systems fail silently
\end{itemize}

This project addresses the following question:

\textit{How can we automatically detect data drift and trigger corrective actions without human intervention?}

\section{Key Features}

\begin{itemize}
\item Feature-wise data drift detection
\item PSI (Population Stability Index) monitoring
\item Model performance monitoring
\item Automatic retraining mechanism
\item Model versioning and rollback
\item Production batch simulation
\item Monitoring dashboard using Streamlit
\item Feature distribution visualization
\end{itemize}

\section{System Architecture}

\begin{center}
\begin{verbatim}

                Reference Dataset
                        |
                        v
                Training Pipeline
                        |
                        v
                Model Registry
                        |
                        v
                  Prediction API
                        |
                        v
             Production Data Batches
                        |
                        v
               Monitoring Pipeline
                        |
        ----------------------------------
        |                                |
        v                                v
  Drift Detection               Performance Monitor
        |                                |
        ----------- Retraining Trigger ---
                        |
                        v
                  New Model Version

\end{verbatim}
\end{center}

\section{Project Folder Structure}

\begin{lstlisting}
Self-Healing-ML-Pipeline-System

src/
  components/
    data_validation.py
    drift_detector.py
    performance_monitor.py
    data_transformation.py
    model_trainer.py

  pipelines/
    training_pipeline.py
    monitoring_pipeline.py
    prediction_pipeline.py

  registry/
    model_registry.py

ui/
  app.py
  pages/
    prediction.py
    monitoring.py
    drift_dashboard.py
    model_registry.py

artifacts/
data/
test_drift_data_maker.py
requirements.txt
\end{lstlisting}

\section{Data Drift Detection}

Two statistical techniques are used.

\subsection{Kolmogorov-Smirnov Test}

Used to compare the distributions of reference and production datasets.

\[
D = \sup_x |F_1(x) - F_2(x)|
\]

Where:

\begin{itemize}
\item $F_1(x)$ = reference distribution
\item $F_2(x)$ = production distribution
\end{itemize}

\subsection{Population Stability Index (PSI)}

PSI measures the magnitude of distribution shift.

\[
PSI = \sum (Actual_i - Expected_i) \times \ln \left(\frac{Actual_i}{Expected_i}\right)
\]

Interpretation:

\begin{center}
\begin{tabular}{|c|c|}
\hline
PSI Value & Interpretation \\
\hline
$< 0.1$ & Stable \\
0.1 -- 0.2 & Moderate Drift \\
$> 0.2$ & Significant Drift \\
\hline
\end{tabular}
\end{center}

\section{Monitoring Strategy}

The monitoring pipeline performs the following steps:

\begin{enumerate}
\item Load latest production batch
\item Validate incoming data
\item Detect feature-wise drift
\item Evaluate model performance
\item Trigger retraining if required
\end{enumerate}

Retraining condition:

\begin{lstlisting}
Drift detected AND Performance degraded
\end{lstlisting}

\section{Dashboard Overview}

The project includes a \textbf{Streamlit monitoring dashboard} with:

\begin{itemize}
\item Prediction interface
\item Monitoring control panel
\item Drift visualization dashboard
\item Model registry viewer
\end{itemize}

\section{How to Run the Project}

\subsection{Install Dependencies}

\begin{lstlisting}
pip install -r requirements.txt
\end{lstlisting}

\subsection{Generate Production Batches}

\begin{lstlisting}
python test_drift_data_maker.py
\end{lstlisting}

This script splits the dataset into simulated production batches.

\subsection{Train Initial Model}

\begin{lstlisting}
python -m src.pipelines.training_pipeline
\end{lstlisting}

\subsection{Launch Dashboard}

\begin{lstlisting}
streamlit run ui/app.py
\end{lstlisting}

\section{Challenges Faced}

\subsection{Histogram Binning Errors}

During PSI calculation, the system encountered:

\begin{lstlisting}
ValueError: bins must increase monotonically
\end{lstlisting}

Solution:

\begin{itemize}
\item Remove NaN values
\item Generate monotonic bin edges using numpy
\end{itemize}

\subsection{Model Registry Synchronization}

Registry occasionally referenced deleted models.

Solution:

\begin{itemize}
\item Added validation checks
\item Implemented fallback loading
\end{itemize}

\section{Technologies Used}

\begin{center}
\begin{tabular}{|c|c|}
\hline
Category & Technology \\
\hline
Language & Python \\
Machine Learning & Scikit-Learn \\
Data Processing & Pandas \\
Visualization & Plotly \\
Dashboard & Streamlit \\
Drift Detection & KS Test + PSI \\
\hline
\end{tabular}
\end{center}

\section{Future Improvements}

\begin{itemize}
\item Integration with MLflow
\item Real-time monitoring pipelines
\item Feature importance drift analysis
\item CI/CD integration
\item Cloud deployment
\end{itemize}

\section{Conclusion}

This project demonstrates a production-style ML monitoring system capable of detecting data drift and automatically retraining models to maintain reliability.

The architecture focuses on:

\begin{itemize}
\item Modular pipeline design
\item Monitoring-first ML engineering
\item Automated model maintenance
\end{itemize}

\end{document}