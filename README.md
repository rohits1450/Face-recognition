 # Face Recognition Using Deep Learning:
  
## Tech Stacks and Technologies Used
- Python
- OpenCV
- DeepFace
- Facenet
- Google Colab

## Main Goal
The main goal of this project is to perform face recognition by comparing known faces with unknown faces. The project utilizes pre-trained deep learning models to identify and match faces in images, providing accurate and efficient face verification.

## Installation
```sh
!pip install opencv-python opencv-python-headless
!pip install deepface

IMPORTING LIBRARIES
The script begins by importing the necessary libraries:

python
import cv2
import os
from deepface import DeepFace
from google.colab.patches import cv2_imshow
OPENCV: Used for image processing and computer vision tasks.

DEEPFACE: Used for deep learning-based face recognition.

GOOGLE COLAB PATCHES: Used to display images within Google Colab notebooks.

READING AND RESIZING IMAGES
A function is defined to read and resize images:

python
def read_img(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load image: {path}")
        return None
    (h, w) = img.shape[:2]
    width = 500
    r = width / float(w)
    height = int(h * r)
    return cv2.resize(img, (width, height))
READ_IMG(): Reads an image from the given path, resizes it to a standard width, and maintains the aspect ratio.

LOADING KNOWN FACES
The script initializes lists for storing known face encodings and names:

python
known_encodings = []
known_names = []
known_dir = 'known'

for file in os.listdir(known_dir):
    if file.startswith('.'):
        continue
    img = read_img(known_dir + '/' + file)
    if img is not None:
        known_encodings.append(img)
        known_names.append(file.split('.')[0])
KNOWN_ENCODINGS: List to store image encodings of known faces.

KNOWN_NAMES: List to store names corresponding to the known faces.

The script iterates through the 'known' directory, reads, and resizes each image, then stores the encodings and names.

PROCESSING UNKNOWN FACES
The script iterates through the 'unknown' directory to process unknown faces:

python
unknown_dir = 'unknown'
for file in os.listdir(unknown_dir):
    if file.startswith('.'):
        continue
    print("Processing", file)
    img = read_img(unknown_dir + '/' + file)
    if img is not None:
        results = []
        for known_face, name in zip(known_encodings, known_names):
            result = DeepFace.verify(img, known_face, model_name='Facenet', distance_metric='cosine', threshold=0.4)
            is_match = result['verified']
            if is_match:
                results.append(name)
UNKNOWN_DIR: Directory containing images of unknown faces.

The script reads and resizes each unknown face image, compares it with known faces using DeepFace, and stores the matched names.

DISPLAYING RESULTS
The script overlays matched names on the unknown face image and displays it:

python
        if results:
            for i, name in enumerate(results):
                cv2.putText(img, name, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2_imshow(img)
        else:
            print("No matches found")

        print(file)
        print("Matched with:", results)
If matches are found, the script uses CV2.PUTTEXT to overlay the matched names on the image and displays it using CV2_IMSHOW.

If no matches are found, it prints a message indicating this.
----------------------------------------------------------------------------------------------------------------------------


