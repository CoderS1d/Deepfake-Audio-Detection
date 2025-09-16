# Deepfake-Audio-Detection
Deepfake Audio Detection

This project focuses on building a Deepfake Audio Detection system capable of identifying manipulated or synthetically generated speech. With the rise of generative AI, deepfake voices pose significant risks in misinformation, security, and fraud. This system leverages wav2vec embeddings and autoencoder-based anomaly detection to reliably distinguish between real and fake audio.

🔑 Key Features

Wav2Vec for Feature Extraction – Used pre-trained wav2vec models to capture rich speech embeddings and subtle acoustic patterns.

Autoencoder for Anomaly Detection – Trained autoencoders to reconstruct genuine audio embeddings and flag anomalies in deepfake audio.

Robust Preprocessing – Applied noise reduction, spectrogram generation, and feature normalization to improve detection accuracy.

Evaluation Metrics – Assessed performance using accuracy, precision, recall, and F1-score on benchmark datasets.

Scalable Pipeline – Designed workflows that can be extended to real-time detection applications.

📂 Tech Stack

Languages/Frameworks: Python, PyTorch, Librosa

Models: Wav2Vec, Autoencoders

Tools: Scikit-learn, Matplotlib, Jupyter

🚀 Applications

Detecting manipulated audio in fraud prevention (e.g., voice phishing).

Ensuring authenticity in media and journalism.

Protecting digital identities from synthetic impersonation.
