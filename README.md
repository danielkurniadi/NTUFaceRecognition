
# Face Recognition Pipelines

The technology of Face Recognition has been employed in many applications in the digital era, from security and surveillance to mobile applications. However, a large amount of data will be stored in a database if there are many faces that need to be recognized, which will lead to an efficiency problem in aspects of memory and time consuming. Thus, we need a robust and compact representation of the face data that allows faster self-similarity matching which is not the case when storing raw images.

Principal component analysis addressed the problem by allowing a more compact and robust representation. This further enables a faster and more robust face matching in a face recognition system. This project follows the general thought of a face recognition system.

### Jupyter Notebook: Docs and Experimentation Results 

Our experimentation most likely take place in Jupyter (Python3) Notebook environment. Our notebooks can be found under
[notebooks folder](notebooks/). The experimentation is the following:

* **Face PCA.ipynb**: Experimentation with PCA, SVM-SGD Classifier for face recognition and emotion classification. Plotting and visualisation included.

* **face_pipeline.ipynb**: Experimentation with training and live demo pipeline with image processing stages

Refer to the official site of [Jupyterlab](https://jupyter.org/) on how to run a jupyter environment, or you can simply
use [Google colabs](https://colab.research.google.com/) as an alternative.

### Example of usage

Just a slightly contrived example for performing training on `EigenFace` model

```python
from facelib import FacePipeline
from facelib.models import EigenFace

# load face dataset + labels into pipeline, see docs regarding data directory structure
face_pipeline = FacePipeline(file_root='<YOUR_PATH_TO_DATADIR>', resize=(300,300)).fit()

# transforms face image into eigenvectors and labels
# if no argument to transforms(), will transform data from file_root
eigenvectors = face_pipeline.transforms()  

# instantiate and train model
model = EigenFace(outlier_thresh=0.8)
model.fit(eigenvectors)  # 2 ez for me, done!

```

Now we can perform testing on a test image. This is as easy as calling the models based from `sklearn.estimators`

```python
# -----------------
# TESTING
# -----------------
import cv2

test_image = cv2.imread('<PATH_TO_TEST>')
eigenvect = face_pipeline.transform(test_image)

prediction = model.predict(eigenvect)  # yield index of class prediction
prediction_name = model.predict(eigenvect, use_name=True)  # yield name of predicted label class
```

## Contributing

Please contribute if you are free <3. Peace out.
