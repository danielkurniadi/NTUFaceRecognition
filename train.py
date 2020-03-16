from dataset import NormalizedDataset, AlignedFaceDataset
from joblib import dump
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid


dataset = NormalizedDataset(face_dataset=AlignedFaceDataset(data_root='new_face_dataset_plus/'))

filenames = []
images = []
labels = []
for i in range(len(dataset)):
    filename = dataset[i][0]

    filenames.append(filename)
    images.append(dataset[i][1])
    labels.append(dataset[2])

image_arr = np.array(images)
pca = PCA()
classifier = NearestCentroid()

x = image_arr
y = list(range(95))  # 95 because excluded one image

x_feat = pca.fit_transform(x)
classifier.fit(x_feat, y)

dump(classifier, 'joint_classif.joblib')
dump(pca, 'joint_pca.joblib')
