from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm

def createdataframe(dir):
  image_paths=[]
  labels=[]
  for label in os.listdir(dir):
    label_path=os.path.join(dir,label)
    if os.path.isdir(label_path):
      for imagename in os.listdir(label_path):
        image_paths.append(os.path.join(label_path,imagename))
        labels.append(label)
        print(label,"completed")
  return image_paths,labels
  
def load_test_images(dir):
  image_paths=[]
  valid_extensions=('.jpg','.png','.jpeg')
  for imagename in os.listdir(dir):
    if imagename.lower().endswith(valid_extensions):
      image_paths.append(os.path.join(dir,imagename))
  return image_paths
  
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(224, 224))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)
    return features
  
TRAIN_DIR = '/content/drive/MyDrive/Data/New_Data'
TEST_DIR = '/content/drive/MyDrive/Data/Test'

train=pd.DataFrame()
train['image'],train['label']=createdataframe(TRAIN_DIR)
train_features=extract_features(train['image'])
x_train=train_features/255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

test_images = load_test_images(TEST_DIR)
test_features = extract_features(test_images)
x_test = test_features / 255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=30, epochs=30)

test_predictions = model.predict(x_test)
test_labels = le.inverse_transform(np.argmax(test_predictions, axis=1))
submission = pd.DataFrame({'Id': [os.path.basename(img) for img in test_images], 'Label': test_labels})
submission.to_csv('vansh.csv', index=False)
df=pd.read_csv('/content/vansh.csv')
df['Label'].value_counts()
