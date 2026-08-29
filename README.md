🤟 Real-Time Sign Language Detection

Real-time ASL fingerspelling recognition using computer vision and machine learning.

A computer vision project that uses a webcam, MediaPipe hand landmarks, and a Random Forest classifier to recognize static ASL fingerspelling signs and convert them into letters.

The project is being developed toward a real-time sign-to-text communication system.

🚧 Project Status

In active development

Current Progress

Webcam hand tracking

21-point hand landmark detection with MediaPipe

Static ASL alphabet dataset collection

Landmark-based feature extraction

Hand landmark normalization

Random Forest classifier

Train/test evaluation

94.80% test accuracy on the current static-sign dataset

Dynamic sequence collection for J and Z

Real-time static-letter prediction

J/Z movement recognition

Real-time sign-to-text interface

Word and sentence formation

Text-to-speech

Multi-user evaluation

Robustness improvements

🎯 Goal

The long-term goal is to build an accessible system that can recognize ASL fingerspelling through a webcam and convert recognized signs into readable text.

Planned Pipeline

                 Webcam
                    │
                    ▼
            Hand Landmark Detection
                    │
                    ▼
              21 Landmarks
                    │
                    ▼
          Feature Normalization
                    │
                    ▼
           Machine Learning Model
                    │
                    ▼
             Sign Recognition
                    │
                    ▼
                  Text
                    │
                    ▼
            Words / Sentences
                    │
                    ▼
             Text-to-Speech

🧠 How It Works

1. Hand Landmark Detection

MediaPipe detects the hand and provides 21 landmarks.

Each landmark contains:

X — horizontal position

Y — vertical position

Z — depth estimate

Therefore, each hand produces:

21 landmarks × 3 coordinates = 63 features

2. Landmark Normalization

Raw coordinates change when the hand moves around the camera frame or changes size.

To make the classifier more robust, the landmarks are normalized by:

Using the wrist landmark as the origin.

Translating the remaining landmarks relative to the wrist.

Scaling the hand according to its overall size.

This helps the model learn the shape and geometry of the hand instead of memorizing its exact location in the frame.

3. Classification

A Random Forest classifier learns the relationship between the normalized landmark features and their corresponding ASL letter.

Hand
  ↓
MediaPipe
  ↓
21 Landmarks
  ↓
Normalization
  ↓
63 Features
  ↓
Random Forest
  ↓
Letter

📊 Model Performance

The current normalized Random Forest model achieved:

94.80% test accuracy

Model

Test Accuracy

Initial Random Forest

82.80%

Normalized Random Forest

94.80%

The improvement came from normalizing the landmark coordinates before training.

Important Note

The 94.80% result is based on the current dataset and held-out test split. It should not be interpreted as 94.80% real-world ASL translation accuracy.

Performance may differ across:

Different users

Cameras

Lighting conditions

Backgrounds

Hand sizes and orientations

Real-world usage

🔤 ASL Fingerspelling

The project currently treats the alphabet in two categories.

Static Signs

Most fingerspelling letters can be represented primarily by a single hand configuration:

A B C D E F G H I
K L M N O P Q R S T U V W X Y

Dynamic Signs

J and Z involve movement, so they are collected as landmark sequences over time rather than as a single static frame.

J → movement sequence
Z → movement sequence

These will be handled by a separate temporal/movement-recognition stage.

This project focuses on ASL fingerspelling rather than attempting to represent the complete ASL language.

📁 Project Structure

RealTime-Sign-Language-Detection/
│
├── images/
│
├── hand_sign.py
│   └── Hand landmark detection
│
├── collect_data.py
│   └── Static sign data collection
│
├── collect_dynamic.py
│   └── Dynamic J/Z sequence collection
│
├── train_model.py
│   └── Static Random Forest training
│
├── hand_sign_dataset.csv
│   └── Static landmark dataset
│
├── dynamic_hand_sign_dataset.csv
│   └── Dynamic landmark sequences
│
├── hand_landmarker.task
│   └── MediaPipe hand landmark model
│
├── asl_static_model.pkl
│   └── Initial classifier
│
├── asl_static_model_normalized.pkl
│   └── Normalized classifier
│
├── face_detection.py
│   └── Earlier OpenCV experiment
│
├── .gitignore
└── README.md

🛠️ Tech Stack

Technology

Purpose

Python

Core programming language

OpenCV

Webcam and image processing

MediaPipe

Hand landmark detection

NumPy

Numerical operations

Scikit-learn

Machine learning

Random Forest

Static sign classification

Joblib

Model serialization

CSV

Dataset storage

⚙️ Installation

1. Clone the repository

git clone https://github.com/Ritu198007/RealTime-Sign-Language-Detection.git
cd RealTime-Sign-Language-Detection

2. Create a virtual environment

python -m venv venv

3. Activate the environment

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

4. Install dependencies

pip install opencv-python mediapipe numpy scikit-learn joblib

▶️ Run Hand Landmark Detection

Run:

python hand_sign.py

This opens the webcam and displays the detected hand landmarks.

The current implementation tracks the 21 landmarks of a detected hand in real time.

📥 Collect Static Sign Data

Run:

python collect_data.py

The collector allows a letter to be selected and records multiple landmark samples.

A useful dataset should contain natural variations such as:

Slight hand-position changes

Small rotations

Different distances from the camera

Natural finger-position differences

Different lighting conditions

For better generalization, future datasets should also include samples from multiple users.

🎥 Collect Dynamic J/Z Data

Run:

python collect_dynamic.py

Unlike static signs, J and Z are represented using sequences of landmarks over time.

Conceptually:

Frame 1
   ↓
Frame 2
   ↓
Frame 3
   ↓
  ...
   ↓
Frame N
   ↓
Movement Sequence
   ↓
J / Z

🤖 Train the Static Model

Run:

python train_model.py

The training pipeline:

Loads the static dataset.

Extracts landmark coordinates.

Normalizes the hand landmarks.

Splits the dataset into training and testing sets.

Trains a Random Forest classifier.

Evaluates the classifier.

Saves the trained model.

The normalized model is saved as:

asl_static_model_normalized.pkl

🔬 Dataset Strategy

The model currently uses landmark data rather than storing full camera images for each sample.

For each static sample:

21 landmarks
×
3 coordinates
=
63 numerical features

The goal is to capture variations while preserving the underlying handshape.

Future Dataset Improvements

The dataset can be expanded with:

More samples per letter

Multiple users

Different camera angles

Different lighting

Different backgrounds

Different hand distances

More natural hand orientations

🔮 Roadmap

Phase 1 — Computer Vision Foundation

Webcam input

Hand detection

Landmark extraction

Phase 2 — Dataset

Static alphabet collection

Landmark normalization

Dynamic J/Z collection

Phase 3 — Machine Learning

Random Forest classifier

Model evaluation

94.80% current test accuracy

Improve generalization

Phase 4 — Real-Time Recognition

Webcam prediction

Confidence filtering

Stable prediction over consecutive frames

J/Z movement recognition

Phase 5 — Sign to Text

Letter accumulation

Word formation

Sentence formation

Correction/backspace mechanism

Text-to-speech

Phase 6 — Real-World Testing

Multi-user testing

Different environments

Performance optimization

User-friendly interface

💡 Example of the Intended Experience

The final application is intended to work approximately like this:

User performs a sign
        ↓
Webcam captures hand
        ↓
MediaPipe extracts landmarks
        ↓
Model recognizes the sign
        ↓
Recognized letter appears
        ↓
Letters form words
        ↓
Words form sentences

Example:

H → E → L → L → O

HELLO

Eventually:

HELLO HOW ARE YOU

can be displayed as text and optionally converted to speech.

⚠️ Limitations

This is currently an experimental prototype.

The model may perform differently depending on:

User

Camera quality

Lighting

Background

Hand orientation

Distance from camera

Similar-looking handshapes

The current 94.80% test accuracy does not guarantee equivalent performance in real-world conditions.

Additionally, recognizing individual fingerspelling letters is different from translating complete ASL conversations. Full sign-language translation requires substantially more linguistic and temporal context.

🚀 Future Improvements

Potential improvements include:

More diverse training data

Multi-user datasets

Better feature engineering

Temporal models for dynamic signs

Prediction smoothing

Confidence thresholds

Deep-learning-based classifiers

Real-time text generation

Text-to-speech

A graphical user interface

Mobile or edge deployment

🤝 Contributing

Contributions, ideas, and suggestions are welcome.

If you'd like to contribute:

Fork the repository.

Create a feature branch.

Make your changes.

Test your changes.

Open a pull request.

📜 Disclaimer

This project is an educational and experimental computer vision project.

It is intended to explore ASL fingerspelling recognition and sign-to-text technology. It should not currently be considered a complete, authoritative, or production-ready ASL translation system.

⭐ Project

Real-Time Sign Language Detection

Built with:

Python · OpenCV · MediaPipe · NumPy · Scikit-learn

Turning hand movements into meaningful text — one sign at a time. 🤟