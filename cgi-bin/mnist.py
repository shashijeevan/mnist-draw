#!/usr/bin/env python
"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""

import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
#from model import model

import keras
from keras.models import model_from_json


# Default output
res = {"result": 0,
       "data": [], 
       "error": ''}

try:
    # Get post data
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

        # Convert data url to numpy array
        img_str = re.search(r'base64,(.*)', data).group(1)
        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        arr = np.array(im)[:,:,0:1]

        # Normalize and invert pixel values
        arr = (255 - arr) / 255.

        img_rows, img_cols = 28, 28

        # Load trained model
#        model.load('cgi-bin/models/model.tfl')

        # Predict class
#        predictions = model.predict([arr])[0]

#Using saved model and weights for prediction

# load json and create model
        json_file = open('cgi-bin/models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
# load weights into new model
        loaded_model.load_weights("cgi-bin/models/model.h5")

        loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        test_img = arr.reshape((1,img_rows,img_cols,1))
        res['data'] = [float(num) for num in loaded_model.predict(test_img)[0]]

        # Return label data
        res['result'] = 1
#        res['data'] = [float(num) for num in predictions] 

except Exception as e:
    # Return error data
    res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


