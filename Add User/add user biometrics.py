import os
from skimage import io
import dlib
import openpyxl as xlsx

upFolderPath = os.path.dirname(os.path.dirname(os.path.abspath('add user biometrics.py')))

def getImageFilename():
    for filename in os.listdir(os.path.abspath('')):
        for ext in ['.jpg', '.jpeg', '.png']:
            if ext in filename.lower():
                return filename
    return None
def getBiometrics(filename):
    sp = dlib.shape_predictor(upFolderPath + '\shape_predictor_68_face_landmarks.dat') # 'shape_predictor_5_face_landmarks.dat'
    face_rec = dlib.face_recognition_model_v1(upFolderPath + '\dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()

    image = io.imread(filename)
    detected_face = detector(image, 1)[0]
    try:
        shape = sp(image, detected_face)
        face_descriptor = face_rec.compute_face_descriptor(image, shape)
        if(face_descriptor != None):
            username = filename.split('.')[:-1]
            bio = username + list(face_descriptor)
            return bio
    except:
        return None
def saveData(data, dbFilemane):
    wb = xlsx.load_workbook(dbFilemane)
    sheet = wb.active
    sheet.append(data)
    wb.save(dbFilemane)

try:
    faceImageFilename = getImageFilename()
    biometrics = getBiometrics(faceImageFilename)
    saveData(biometrics, upFolderPath + '\\data.xlsx')
    print('Success!')
except Exception as ex:
    print('Error: '+ str(ex))