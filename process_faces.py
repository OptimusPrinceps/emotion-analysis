import os
import pandas as pd
import matplotlib.pyplot as plt 
from fer import FER

OVERWRITE = False
data_filename = 'not_me_emotion_data.csv'
image_dir = './notmyface'


if OVERWRITE or not os.path.exists(data_filename):
    with open(data_filename, 'w') as f:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        f.write(','.join(['filename', 'date_created', 'box_x', 'box_y', 'box_width', 'box_height', *emotions]) + '\n')


emotion_data = pd.read_csv(data_filename)

existing_files = set(emotion_data.filename)

faces = os.listdir(image_dir)
detector = FER(mtcnn=True)

for i, image_name in enumerate(faces):
    if image_name in existing_files:
        continue

    img_path = os.path.join(image_dir, image_name)

    # try:
    #     img_metadata = Image(img_path)
    #     date_taken = img_metadata.get('datetime_original')
    # except:
    #     date_taken = 'None'

    try:
        img = plt.imread(img_path)
        image_faces = detector.detect_emotions(img)
    except:
        continue

    f = open(data_filename, 'a')
    for face in image_faces:
        row = [image_name, '', *face['box'], *face['emotions'].values()]
        f.write(','.join([str(x) for x in row]) + '\n')
    
    f.close()
        
    if i % 20 == 0:
        print(f'Processed {i}/{len(faces)} images')
