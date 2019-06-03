
from __future__ import absolute_import

import echolib
import echocv

import math
import cv2

from tornado.ioloop import IOLoop
import tornado.web

def install_client(ioloop, client, disconnect_callback=None):
    def _tornado_handler(fd, events):
        if events & IOLoop.ERROR:
            client.disconnect()
            if disconnect_callback:
                disconnect_callback(client)
        else:
            if events & IOLoop.READ:
                client.handle_input()
            if events & IOLoop.WRITE:
                client.handle_output()

    ioloop.add_handler(client.fd(), _tornado_handler, IOLoop.READ | IOLoop.WRITE | IOLoop.ERROR | IOLoop._EPOLLET)

def uninstall_client(ioloop, client):
    ioloop.remove_handler(client.fd())

class Image(object):
    def __init__(self, frame):
        self._raw = frame.image
        self._timestamp = frame.header.timestamp
        self._jpeg = None

    def raw(self):
        return self._raw

    def timestamp(self):
        return self._timestamp

    def jpeg(self):
        if self._jpeg is None:
            result, data = cv2.imencode('.jpg', self._raw, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if result == False:
                return None
            self._jpeg = str(data.data)
        return self._jpeg

class Camera(object):
    def __init__(self, client, name):
        self.name = name
        self._image_listeners = []
        self._location_listeners = []
        self._image = None
        self._client = client
        self._parameters = echocv.CameraIntrinsicsSubscriber(client, "%s.parameters" % name, lambda x: self._parameters_callback(x))
        self._location = None

    def listen_images(self, listener):
        self._image_listeners.append(listener)
        if len(self._image_listeners) == 1 and self._image is None:
            self._image = echocv.FrameSubscriber(self._client, "%s.image" % self.name, lambda x: self._frame_callback(x))

    def unlisten_images(self, listener):
        self._image_listeners.remove(listener)
        if len(self._image_listeners) == 0 and not self._image is None:
            self._image = None

    def listen_location(self, listener):
        self._location_listeners.append(listener)
        if len(self._location_listeners) == 1 and self._location is None:
            self._location = echocv.CameraExtrinsicsSubscriber(self._client, "%s.location" % self.name, lambda x: self._location_callback(x))

    def unlisten_location(self, listener):
        self._location_listeners.remove(listener)
        if len(self._location_listeners) == 0 and not self._location is None:
            self._location = None

    def _distribute_image(self, image):       
        for c in self._image_listeners:
            c.push_image(image)
            
    def _distribute_location(self, location):       
        for c in self._location_listeners:
            c.push_camera_location(self, location)

    def _frame_callback(self, frame):
        if len(self._image_listeners) > 0:
            img = Image(frame)
            self._distribute_image(img)

    def _location_callback(self, location):
        self._distribute_location(location)

    def _parameters_callback(self, parameters):
        self.parameters = parameters

class VideoHandler(tornado.web.RequestHandler):

    def __init__(self, application, request, camera):
        super(VideoHandler, self).__init__(application, request)
        self.camera = camera
        self.flushing = False

    @tornado.web.asynchronous
    def get(self):
        # TODO: add random chars to binary
        self.boundary = '--imageboundary--'
        self.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=' + self.boundary)
        self.flush()
        self.camera.listen_images(self)

    def push_image(self, image):
        if len(self.boundary) < 1 or self.flushing:
            return
        
        data = image.jpeg()

        self.write(self.boundary)
        self.write("\r\n")
        self.write("Content-type: image/jpeg\r\n")
        self.write("Content-length: %d\r\n\r\n" % len(data))
        self.write(data)
        self.write("\r\n")
        self.flushing = True
        self.flush(callback=lambda: self.on_flush_complete())

    def on_flush_complete(self):
        self.flushing = False

    def on_finish(self):
        self.camera.unlisten_images(self)

    def on_connection_close(self):
        self.camera.unlisten_images(self)

    def check_etag_header(self):
        return False

class ImageHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, camera):
        super(ImageHandler, self).__init__(application, request)
        self.camera = camera

    def set_default_headers(self):
        self.set_header('Content-Type', 'image/jpeg')
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')

    @tornado.web.asynchronous
    def get(self):
        self.camera.listen_images(self)

    def push_image(self, image):
        self.set_header('X-Timestamp', image.timestamp().isoformat())
        self.write(image.jpeg())
        self.finish()

    def on_finish(self):
        self.camera.unlisten_images(self)

    def on_connection_close(self):
        self.camera.unlisten_images(self)

    def check_etag_header(self):
        return False


__all__ = ["Camera", "ImageHandler", "VideoHandler", "install_client", "uninstall_client"]