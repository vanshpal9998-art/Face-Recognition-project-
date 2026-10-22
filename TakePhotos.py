import os
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

if os.path.exists('id-names.csv'):
    id_names = pd.read_csv('id-names.csv')
    id_names = id_names[['id', 'name']]
else:
    id_names = pd.DataFrame(columns=['id', 'name'])
    id_names.to_csv('id-names.csv')

if not os.path.exists('faces'):
    os.makedirs('faces')

print('Welcome!')
print('\nPlease put in your ID.')
print('If this is your first time choose a random ID between 1-10000')

id = int(input('ID: '))
name = ''

if id in id_names['id'].values:
    name = id_names[id_names['id'] == id]['name'].item()
    print(f'Welcome Back {name}!!')
else:
    name = input('Please Enter you name: ')
    os.makedirs(f'faces/{id}')
    id_names.loc[len(id_names)] = {'id': id, 'name': name}
    id_names.to_csv('id-names.csv', index=False)

print("\nLet's capture!")

print("Now this is where you begin taking photos. Once you see a rectangle around your face, press the 's' key to capture a picture.", end=" ")
print("It is recommended to take atleast 20-25 pictures, from different angles, in different poses, with and without specs, you get the gist.")
input("\nPress ENTER to start when you're ready, and press the 'q' key to quit when you're done!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, 'Classifiers', 'haarface.xml')

camera = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(xml_path)

photos_taken = 0

while(cv2.waitKey(1) & 0xFF != ord('q')):
    _, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        face_region = grey[y:y + h, x:x + w]
        if cv2.waitKey(1) & 0xFF == ord('s') and np.average(face_region) > 50:
            face_img = cv2.resize(face_region, (220, 220))
            img_name = f'face.{id}.{datetime.now().microsecond}.jpeg'
            cv2.imwrite(f'faces/{id}/{img_name}', face_img)
            photos_taken += 1
            print(f'{photos_taken} -> Photos taken!')

    cv2.imshow('Face', img)

camera.release()
cv2.destroyAllWindows()
 