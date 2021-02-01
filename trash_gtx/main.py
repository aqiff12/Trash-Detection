from ctypes import *
import math
import random
import os
import numpy as np
import cv2
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class darknet_main():
    def __init__(self):
        self.netMain = None
        self.metaMain = None
        self.altNames = None

        self.thresh = 0.5

        self.configPath = './model/yolov3_custom_fyp_testing.cfg'
        self.weightPath = './model/yolov3_custom_fyp_last.weights'
        self.metaPath = "./model/obj_fyp.data"
        self.frame = None

    def setGPU(self, is_GPU):
        self.hasGPU = is_GPU
        if self.hasGPU:
            self.lib = CDLL("yolo_cpp_dll.dll", RTLD_GLOBAL)
        else:
            self.lib = CDLL("yolo_cpp_dll_nogpu.dll", RTLD_GLOBAL)

        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        if self.hasGPU:
            self.set_gpu = self.lib.cuda_set_device
            self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                           POINTER(c_int),
                                           c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)


        self.netMain = self.load_net_custom(self.configPath.encode("ascii"), self.weightPath.encode("ascii"), 0,
                                            1)  # batch size = 1
        self.metaMain = self.load_meta(self.metaPath.encode("ascii"))



        try:
            with open(self.metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            self.namesList = namesFH.read().strip().split("\n")
                            self.altNames = [x.strip() for x in self.namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    def sample(self, probs):
        s = sum(probs)
        probs = [a/s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs)-1

    def c_array(self, ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    def array_to_image(self, arr):
        import numpy as np
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w,h,c,data)
        return im, arr

    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            if self.altNames is None:
                nameTag = meta.names[i]
            else:
                nameTag = self.altNames[i]
            res.append((nameTag, out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, net, meta, cv_im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
        # im = self.load_image(image, 0, 0)
        # debug=True
        custom_image = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
        h, w, c_ = custom_image.shape
        custom_image = cv2.resize(custom_image,(self.lib.network_width(net), self.lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
        im, arr = self.array_to_image(custom_image)		# you should comment line below: free_image(im)
        if debug: print("Loaded image")
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self.predict_image(net, im)
        if debug: print("did prediction")
        dets = self.get_network_boxes(net, w, h, self.thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
        # dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, meta.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on "+str(j)+" of "+str(num))
            if debug: print("Classes: "+str(meta), meta.classes, meta.names)
            for i in range(meta.classes):
                if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = self.meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self.free_detections(dets, num)
        if debug: print("freed detections")
        return res

    def performDetect(self, cv_img):

        # Do the detection
        detections = self.detect(self.netMain, self.metaMain, cv_img, self.thresh)

        imcaptions = []
        boundingBoxs=[]
        for detection in detections:
            label = detection[0]
            confidence = detection[1]
            pstring = label+": "+str(np.rint(100 * confidence))+"%"
            imcaptions.append(pstring)
            print(pstring)
            bounds = detection[2]

            yExtent = int(bounds[3])
            xEntent = int(bounds[2])
            # Coordinates are around the center
            xCoord = int(bounds[0] - bounds[2]/2)
            yCoord = int(bounds[1] - bounds[3]/2)
            boundingBox = [
                (xCoord, yCoord),
                (xCoord, yCoord + yExtent),
                (xCoord + xEntent, yCoord + yExtent),
                (xCoord + xEntent, yCoord)
            ]
            boundingBoxs.append(boundingBox)
            # cv_img = cv2.rectangle(cv_img, boundingBox[0], boundingBox[2], (0,0, 255), 1)
        #
        # cv2.imshow("image", cv_img)
        # cv2.waitKey(0)
        return imcaptions, boundingBoxs

if __name__ == "__main__":
    darknetmain = darknet_main()
    darknetmain.setGPU(is_GPU=False)
    imagePath = "./data/bottle_8.png"
    cv_img = cv2.imread(imagePath)
    darknetmain.performDetect(cv_img)
