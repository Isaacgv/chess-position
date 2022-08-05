import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pickle
from skimage.util.shape import view_as_blocks


model = tf.keras.models.load_model("fen_model")
with open('encoder.pkl', 'rb') as handle:
    enc = pickle.load(handle)

piece_symbols = np.array(list('prbnkqPRBNKQF'))



def labels_from_fen(fen_code):
    '''
    Encoded label data fraom FEN description
    
    '''
  
    fen_code = fen_code.split('.')[0]
    fen_code = fen_code.split('-')
    
    temp = []
    
    for line in fen_code:
        for n in line:
            if n in piece_symbols :
                temp.append(enc.transform([n])[0])
            else:
                for space in range(int(n)):
                    temp.append(enc.transform(['F'])[0])

    return temp

def split_chessboard(image):
    '''
    Split each chessboard image into 64 images
    
    '''
    #img_read = cv2.imread(path + image_path, cv2.IMREAD_GRAYSCALE)
    img_read = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blocks = view_as_blocks(img_read, block_shape=(50, 50))
    return blocks.reshape(-1, 50,50)

        
def pred_generator(images, y_label=True):
    '''
    Separate data for testing or validation
    
    '''
    for i, img in enumerate(images):
        
        y = None
        
        if y_label:
            y = np.array(labels_from_fen(img))
        
        #data normalization
        x = split_chessboard(img)/255.
        
        yield x, y


def predict_fen(image):
    '''
    Return FEN code
    '''
    result = model.predict(list(pred_generator([image], y_label=False))[0][0]).argmax(axis=-1)

    label_enc = enc.inverse_transform(result).reshape(8,-1)

    fen_code = ""
    for line in label_enc:
        count = 0
        for label in line:
            if label =='F':
                count+=1
            else:
                if count != 0:
                    fen_code += str(count)
                    fen_code += label
                    count = 0
                else:
                    fen_code += label
        if count != 0:
            fen_code += str(count)

        fen_code += '-'
    print("Predict FEN: {}".format(fen_code[:-1]))
    return fen_code[:-1]


# Load image.
uploaded_file = st.sidebar.file_uploader("Upload an image:")

if uploaded_file is not None:

    #convert string data to numpy array
    npimg = np.fromstring(uploaded_file.getvalue(), np.uint8)
    # convert numpy array to image
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    st.image(frame)
    result = predict_fen(frame)
    st.write("FEN code:", result)