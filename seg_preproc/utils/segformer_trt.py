import os
import cv2
import numpy as np
import json

# TensorRT dependencies
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class SegFormerTRT:
    def __init__(self, trt_engine_path, t_path=None, dtype=np.float32) -> None:
        self.engine = self.load_trt_engine(trt_engine_path)
        self.cuda_ctx = pycuda.autoinit.context
        self.stream = cuda.Stream()

        self.dtype = dtype
        self.input_dims = self.engine.get_binding_shape(0)
        self.output_dims = self.engine.get_binding_shape(1)
        self.d_input, self.d_output = self.allocate_buffers()

        self.shape = None
        self.taxonomy = self.load_taxonomy(t_path)

    def __del__(self):
        del self.d_input
        del self.d_output
        del self.stream

    @staticmethod
    def load_trt_engine(path):
        with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    @staticmethod
    def load_taxonomy(path):
        with open(os.path.expanduser(path), 'rb') as file:
            return json.load(file)

    def allocate_buffers(self):
        d_input = cuda.mem_alloc(trt.volume(self.input_dims) * np.dtype(self.dtype).itemsize)
        d_output = cuda.mem_alloc(trt.volume(self.output_dims) * np.dtype(self.dtype).itemsize)
        return d_input, d_output

    def inference(self, input_data):
        self.cuda_ctx.push()

        context = self.engine.create_execution_context()

        input_data = np.ascontiguousarray(input_data)
        # Copy data from host to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        # Run inference
        context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)],
                                 stream_handle=self.stream.handle)
        # Copy data from device to host
        output_data = np.empty(self.output_dims, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)

        self.stream.synchronize()
        self.cuda_ctx.pop()
        del context

        return output_data

    def preprocess_input(self, img):
        self.shape = img.shape[0:2][::-1]
        img = cv2.resize(img, self.input_dims[-2:], interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # CHW format & add batch dimension
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        return img

    def postprocess_output(self, output_data, shape=None):
        pred_seg = output_data.argmax(axis=1)[0]
        return cv2.resize(pred_seg.astype('uint8'), shape if shape else self.shape, interpolation=cv2.INTER_NEAREST)

    def visualize_output(self, img, seg):
        cvt_seg = np.array(self.taxonomy['dms46'])[seg]    # TODO: If path is None
        fig = np.array(self.taxonomy['srgb_colormap'])[cvt_seg]
        objs = np.unique(np.array(self.taxonomy['dms46'])[seg])

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        patches = [mpatches.Patch(color=np.array(self.taxonomy['srgb_colormap'][i]) / 255.,
                                  label=self.taxonomy['shortnames'][i]) for i in objs]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='small')
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax2.imshow(fig, interpolation='none')

        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image