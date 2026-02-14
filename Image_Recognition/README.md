## Face Recognition Photo Filter 📸

A deep learning-based face recognition system to automatically filter and organize photos containing specific individuals from a large photo collection.

## 🎯 Project Overview

This project uses pre-trained deep learning models to identify and separate photos containing a specific person (yourself) from a collection of images. Built using transfer learning with face_recognition library (based on dlib's ResNet-34 model).

## 🧠 Deep Learning Concepts Used

- **Convolutional Neural Networks (CNN)** - For face detection
- **ResNet-34 Architecture** - For face feature extraction
- **Transfer Learning** - Using pre-trained models instead of training from scratch
- **Metric Learning** - Distance-based face matching using embeddings
- **Feature Extraction** - Converting faces to 128-dimensional embeddings

## ✨ Features

- ✅ Automated face detection in images
- ✅ Face recognition using deep learning embeddings
- ✅ Batch processing of multiple images
- ✅ Configurable similarity threshold
- ✅ Comprehensive result analysis and visualization
- ✅ Distance distribution plots
- ✅ Detailed logging and reporting

## 📁 Project Structure
```
image_recognition_project/
├── dataset/
│   └── sample_100/              # 100 test images
├── reference_faces/             # 5-10 reference photos of target person
├── output/
│   ├── matched/                 # Photos containing target person
│   ├── not_matched/             # Photos without target person
│   └── review/                  # Borderline cases for manual review
├── logs/
│   ├── results.csv              # Detailed results
│   └── summary.json             # Summary statistics
├── notebooks/
│   └── face_recognition.ipynb   # Main Colab notebook
├── results/
│   ├── sample_results/          # Sample output images
│   └── visualizations/          # Charts and graphs
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Google Colab account (for running the notebook)
- Google Drive (for storing images)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-photo-filter.git
cd face-recognition-photo-filter
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Libraries
```python
face_recognition
opencv-python
pillow
numpy
pandas
matplotlib
tqdm
```

## 📖 Usage

### Google Colab

1. Open the notebook in Google Colab:
   - Upload `notebooks/face_recognition.ipynb` to Google Colab
   - Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/face-recognition-photo-filter/blob/main/notebooks/face_recognition.ipynb)

2. Mount your Google Drive and set up the folder structure

3. Upload your reference photos (5-10 clear photos of yourself)

4. Upload test images to `dataset/sample_100/`

5. Run all cells sequentially

6. Check results in the `output/` folders

## ⚙️ Configuration

Key parameters you can tune:

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `tolerance` | 0.6 | Similarity threshold for matching | 0.4 - 0.7 |
| `model` | 'hog' | Face detection model | 'hog', 'cnn' |
| `upsample` | 0 | Upsampling for better small face detection | 0 - 2 |

### Adjusting Tolerance
```python
tolerance = 0.6  # Default

# More strict (fewer false positives)
tolerance = 0.5

# More lenient (catches more matches)
tolerance = 0.7
```

## 📊 Results

### Sample Dataset Statistics

- **Total Images Processed**: 100
- **Images with Faces Detected**: 85
- **Matched (containing target person)**: 23
- **Not Matched**: 60
- **Review Required**: 2
- **Processing Time**: ~45 seconds (on Colab)

### Accuracy Metrics

- **Precision**: 95.8% (22 true positives / 23 classified as matched)
- **Recall**: 91.7% (22 true positives / 24 actual photos with target)
- **F1-Score**: 93.7%

## 🔬 How It Works

### 1. Face Detection
```python
face_locations = face_recognition.face_locations(image, model="hog")
```
Uses Histogram of Oriented Gradients (HOG) or CNN to detect faces in images.

### 2. Face Encoding
```python
face_encodings = face_recognition.face_encodings(image, face_locations)
```
Converts each detected face into a 128-dimensional embedding using ResNet-34.

### 3. Face Comparison
```python
distances = face_recognition.face_distance(reference_encodings, face_encoding)
min_distance = np.min(distances)
```
Calculates Euclidean distance between embeddings. Lower distance = more similar faces.

### 4. Classification
```python
if min_distance < tolerance:
    classification = "matched"
elif min_distance < tolerance + 0.1:
    classification = "review"
else:
    classification = "not_matched"
```

## 📈 Visualizations

The project generates several visualizations:

1. **Distance Distribution Histogram** - Shows how face distances are distributed
2. **Box Plot by Classification** - Compares distances across matched/not matched categories
3. **Classification Pie Chart** - Overall classification breakdown
4. **Sample Results Grid** - Visual display of matched and unmatched photos

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ Transfer learning with pre-trained deep learning models
- ✅ Face detection and recognition using CNNs
- ✅ Feature extraction and embedding generation
- ✅ Distance-based similarity metrics
- ✅ Batch processing and data pipeline creation
- ✅ Result analysis and visualization
- ✅ Hyperparameter tuning

## 🔧 Hyperparameter Tuning

The project includes functionality to tune key parameters:
```python
# Test different tolerance values
tolerance_values = [0.4, 0.5, 0.6, 0.7]

# Compare HOG vs CNN detection
models = ['hog', 'cnn']

# Test upsampling levels
upsample_levels = [0, 1, 2]
```

Results show tolerance=0.6 with HOG model provides the best speed/accuracy trade-off.

## 🚧 Limitations

- Requires clear, well-lit reference photos for best accuracy
- Performance degrades with profile views or occluded faces
- CNN model requires GPU for real-time processing
- Not tested on video streams (images only)

## 🔮 Future Improvements

- [ ] Add support for multiple target persons
- [ ] Implement real-time video processing
- [ ] Add web interface for easier usage
- [ ] Fine-tune model on custom dataset
- [ ] Add face clustering for unknown individuals
- [ ] Support for batch download from cloud storage

## 📚 References

- [face_recognition library](https://github.com/ageitgey/face_recognition)
- [dlib C++ Library](http://dlib.net/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**[Kaushlendra Kumar Verma]**
- GitHub: [Kaushlendra Verma](https://github.com/kkvhub)
- LinkedIn: [Kaushlendra Kumar Verma](www.linkedin.com/in/kaushlendra-kumar-verma)
- Email: kkv1@arizona.edu

## 🙏 Acknowledgments

- Built as part of deep learning coursework
- Inspired by Google Photos' face recognition feature

## 📧 Contact

For questions or feedback, please open an issue or reach out via email.

---

**Note**: This project was developed for educational purposes to understand face recognition and deep learning concepts. For production use, consider additional security and privacy measures.

'''

