#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        
        self.plugin = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.input_blob = None
        self.output_blob = None
        self.net = None
        

    def load_model(self, model, device = "CPU", cpu_extension = None, plugin = None):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device=device)
            #self.plugin = IEPlugin()
        else:
            self.plugin = plugin
        
        
        self.core = IECore()
        if cpu_extension and "CPU" in device:
            self.plugin.add_cpu_extension(cpu_extension)
        
        self.net = IENetwork(model = model_xml, weights = model_bin)
        
        ### TODO: Check for supported layers ###
        """supported_layers = self.plugin.query_network(network = self.net, device ="CPU")
        ## Check for unsupported layers
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)"""
        
        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)
            unsupported_layers = \
                [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                exit(1)
        ### TODO: Add any necessary extensions ###
        ###if cpu_extension and "CPU" in device:
           ### self.plugin.add_extension(cpu_extension, device)

        ### TODO: Return the loaded inference plugin ###
        self.net_plugin = self.plugin.load(self.net)
        ### Note: You may need to update the function parameters. ###
        ## Getting input layers
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))

        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=0, inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        status = self.net_plugin.requests[0].wait(-1)
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        output = self.net_plugin.requests[0].outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###
        return output
