import numpy as np
import os
import tensorflow as tf
import pathlib
import base64
import json

from tensorflow import keras
from tensorflow.keras.models import Sequential
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

HOST_ADDRESS = ""
HOST_PORT = 8000

model = tf.keras.models.load_model('saved_model/my_model')
class_names = [dI for dI in os.listdir('flower_photos') if os.path.isdir(os.path.join('flower_photos',dI))]
print(class_names)

img_height = 180
img_width = 180

class RequestHandler(BaseHTTPRequestHandler):

    def send_response(self, code, message=None):
        # self.log_request(code)
        self.send_response_only(code)
        self.send_header('Server','python3 http.server Development Server')     
        self.send_header('Date', self.date_time_string())
        self.send_header('plant', message)
        self.end_headers()  

    def do_POST(self):
        
        if self.path == '/control':
            
            content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
            print(content_length)
            post_data = str(self.rfile.read(content_length).decode('utf-8'))
           
            image = Image.open(BytesIO(base64.b64decode(json.loads(post_data)["image"]))).resize((img_width,img_height))

            predictions = model.predict(tf.expand_dims(keras.preprocessing.image.img_to_array(image), 0))
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

            self.send_response(200, message=class_names[np.argmax(score)])
      
 
def run(server_class=HTTPServer, handler_class=BaseHTTPRequestHandler):
    server_address = (HOST_ADDRESS, HOST_PORT)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

if __name__ == '__main__':
    run(handler_class=RequestHandler)

