"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

# Drawing bounding boxes around the persons detceted

def bounding_boxes(frame, result, args, width, height):
    """bounding boxes on the frames"""
    current_count = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= args.prob_threshold: # taking confidance thershold 0.5 for testing
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (20, 255, 0), 2)
            current_count += 1
            #cv2.putText(frame, str(conf)[0:4], (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    n, c, h, w = net_input_shape

    ### TODO: Handle the input stream ###
    single_image_mode = 0
    #variables for person count
    prev_count = 0
    total_count = 0
    duration = 0
    start_timer = 0
    
    
    if args.input == 'CAM':
        args.input = 0
        
    elif args.input.endswith('.jpeg') or args.input.endswith('png'):
        single_image_mode = 1
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    # Getting input shape
    width = int(cap.get(3))
    height = int(cap.get(4))
    

    ### TODO: Loop until stream is over ###
    
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        
        p_frame = cv2.resize(frame, (w, h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        start_time = time.time()
        infer_network.exec_net(p_frame)
        

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            end_time = time.time()
            #result = plugin.extract_output()
            ### TODO: Get the results of the inference request ###
            diff_time = end_time - start_time
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            frame, current_count = bounding_boxes(frame, result, args, width, height)
            
            ## check if output is ok
            out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width, height))

            
            inf_time = "Inference time: {:.3f}ms".format(diff_time * 1000)
            cv2.putText(frame, inf_time, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 255, 0), 1)
            if current_count < prev_count:
                duration = int(time.time() - start_timer)
                client.publish("person/duration", json.dumps({"duration": duration}))
            if current_count > prev_count:
                start_timer = time.time()
                total_count += current_count - prev_count
                client.publish("person", json.dumps({"total": total_count}))
            if current_count < prev_count:
                duration = int(time.time() - start_timer)
                client.publish("person/duration", json.dumps({"duration": duration}))
            ## TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
         
            client.publish("person", json.dumps({"count": current_count}))
            prev_count = current_count
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        
        if single_image_mode == 1:
            cv2.imwrite('output_image.jpeg', frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
