# Real-Time Sign Language Detection 🤟

A computer vision and machine learning project that aims to recognize ASL (American Sign Language) hand signs in real time and convert them into text.

The project uses a webcam to detect hand landmarks, extract hand features, and classify static ASL alphabet signs using machine learning.

## 🚧 Project Status

**Currently in development.**

### Completed
- [x] Webcam hand detection
- [x] 21-point hand landmark detection using MediaPipe
- [x] ASL alphabet dataset collection
- [x] Landmark-based feature extraction
- [x] Hand landmark normalization
- [x] Random Forest classification
- [x] Model evaluation
- [x] 94.8% test accuracy on the current static-sign dataset
- [x] Dynamic data collection for J and Z

### Next Steps
- [ ] Real-time ASL letter prediction
- [ ] Improve classification accuracy
- [ ] Handle J and Z using movement recognition
- [ ] Build real-time sign-to-text output
- [ ] Add word and sentence formation
- [ ] Add text-to-speech
- [ ] Test with multiple users
- [ ] Improve robustness under different lighting and backgrounds

---

## 🎯 Project Goal

The long-term goal is to develop a real-time system that can recognize sign language through a webcam and convert it into readable text.

The planned pipeline is:

```text
Webcam
   ↓
Hand Detection
   ↓
21 Hand Landmarks
   ↓
Feature Normalization
   ↓
Machine Learning Model
   ↓
ASL Sign Recognition
   ↓
Text

🧠 How It Works
1. Hand Detection

MediaPipe detects the user's hand and provides 21 landmarks for each hand.

Each landmark contains:

X coordinate
Y coordinate
Z coordinate

This gives:

21 landmarks × 3 coordinates = 63 features
2. Feature Normalization

Raw coordinates depend on the position and size of the hand in the camera frame.

To make the model more robust, the landmarks are normalized by:

Using the wrist as the origin.
Translating the remaining landmarks relative to the wrist.
Scaling the coordinates based on hand size.

This allows the model to focus more on the shape of the hand rather than its exact position in the frame.

3. Machine Learning

A Random Forest classifier is trained using the normalized landmark features.

The current static-sign pipeline is:

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
Letter Prediction

📊 Current Model Performance

The current normalized Random Forest model achieved:

94.80% test accuracy

This was an improvement over the initial model:

| Model                    |   Accuracy |
| ------------------------ | ---------: |
| Initial Random Forest    |     82.80% |
| Normalized Random Forest |     94.80% |

The improvement came from normalizing the hand landmarks before training.

Note: The reported accuracy is based on the current dataset and test split. It should not be interpreted as real-world accuracy across different people, cameras, lighting conditions, or backgrounds.

🔤 ASL Alphabet

The current static-sign model is designed around the ASL fingerspelling alphabet.

Static signs:

A B C D E F G H I
K L M N O P Q R S T U V W X Y
Dynamic Signs

The letters J and Z involve hand movement, so they are being handled separately using temporal/movement data rather than treating them as ordinary static hand poses.

J → Movement recognition
Z → Movement recognition

📁 Project Structure

RealTime-Sign-Language-Detection/
│
├── images/
│
├── hand_sign.py
│   └── Hand landmark detection demo
│
├── collect_data.py
│   └── Collects static ASL sign data
│
├── collect_dynamic.py
│   └── Collects movement sequences for dynamic signs
│
├── train_model.py
│   └── Trains the static sign classifier
│
├── hand_sign_dataset.csv
│   └── Static hand landmark dataset
│
├── dynamic_hand_sign_dataset.csv
│   └── Dynamic sign sequence dataset
│
├── hand_landmarker.task
│   └── MediaPipe hand landmark model
│
├── asl_static_model.pkl
│   └── Initial static classifier
│
├── asl_static_model_normalized.pkl
│   └── Normalized static classifier
│
├── face_detection.py
│   └── Earlier OpenCV experiment
│
└── README.md

🛠️ Technologies Used
Python
OpenCV
MediaPipe
NumPy
Scikit-learn
Joblib
CSV

⚙️ Installation

Clone the repository:

git clone https://github.com/Ritu198007/RealTime-Sign-Language-Detection.git

Move into the project directory:

cd RealTime-Sign-Language-Detection

Create a virtual environment:

python -m venv venv

Activate it.

Windows
venv\Scripts\activate
Linux/macOS
source venv/bin/activate

Install the dependencies:

pip install opencv-python mediapipe numpy scikit-learn joblib
▶️ Running Hand Landmark Detection

Run:

python hand_sign.py

This opens the webcam and displays the detected hand landmarks.

The system tracks the 21 landmarks of the hand in real time.

📥 Collecting Static Sign Data

To collect training data:

python collect_data.py

The program allows you to select a letter and collect multiple samples.

Each sample contains the 21 hand landmarks.

For example:

A → 50 samples
B → 50 samples
C → 50 samples
...

Natural variations are useful during data collection, including:

Slight changes in hand position
Slight changes in hand rotation
Different distances from the camera
Natural differences in finger positioning
🎥 Collecting Dynamic Sign Data

J and Z involve movement, so they require sequences of landmarks rather than a single frame.

Run:

python collect_dynamic.py

The program records multiple frames for each movement sequence.

Conceptually:

Frame 1
   ↓
Frame 2
   ↓
Frame 3
   ↓
...
Frame N
   ↓
Movement sequence
   ↓
J / Z
🤖 Training the Model

Run:

python train_model.py

The program:

Loads the dataset.
Extracts landmark coordinates.
Normalizes the hand landmarks.
Splits the dataset into training and testing sets.
Trains a Random Forest classifier.
Evaluates the model.
Saves the trained model.

The normalized model is saved as:

asl_static_model_normalized.pkl
📈 Improving the Model

The current model achieved 94.8% test accuracy, but there is still room for improvement.

Potential improvements include:

More training data

Collect more samples for difficult letters.

Multiple users

Training with different people's hands can improve generalization.

Person 1
Person 2
Person 3
Person 4
        ↓
Larger dataset
        ↓
More robust model
Different environments

Collecting samples under different:

Lighting conditions
Backgrounds
Camera positions
Hand orientations

can make the system more robust.

Better feature engineering

Additional geometric features could be explored, such as:

Finger angles
Distances between landmarks
Relative finger lengths
Joint angles
🔮 Future Features

The long-term goal is to turn the classifier into a complete communication application.

Real-Time Prediction
Webcam
 ↓
Hand landmarks
 ↓
Model
 ↓
Predicted letter

Example:

Prediction: A
Confidence: 96%
Word Formation

Individual predictions can be combined:

H → E → L → L → O

HELLO
Sentence Formation

The system can eventually support:

HELLO HOW ARE YOU
Text-to-Speech

The generated text can eventually be converted into speech so that the system can communicate the signed message verbally.

⚠️ Limitations

This project is currently a prototype.

The model's performance may vary depending on:

User
Camera quality
Lighting
Background
Hand orientation
Distance from camera
Similar-looking hand signs

The current 94.8% accuracy is based on the current dataset and test split and does not guarantee the same performance in real-world usage.

🤝 Future Development

The project is being developed incrementally:

Hand Detection
      ↓
Landmark Extraction
      ↓
Dataset Collection
      ↓
Feature Normalization
      ↓
Model Training
      ↓
Real-Time Prediction
      ↓
J/Z Movement Recognition
      ↓
Sign → Text
      ↓
Sentence Formation
      ↓
Text → Speech
📜 Disclaimer

This project is an educational and experimental computer vision project.

It is intended to explore real-time hand-sign recognition and should not be considered a complete or authoritative ASL translation system.

⭐ Project

Real-Time Sign Language Detection

Built with Python, OpenCV, MediaPipe, and Machine Learning.


### One thing I'd change before committing

Your README currently says **94.8%**, which is great, but keep the wording **"94.8% test accuracy on the current dataset"** rather than just "94.8% accuracy." That's much more credible on GitHub because it makes clear this isn't a claim of 94.8% real-world ASL translation accuracy.

Then commit it:

```bash
git add README.md
git commit -m "Add project documentation"
git push