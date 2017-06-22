
from __future__ import absolute_import

import echolib
import echocv

import cv2

from tornado.ioloop import IOLoop
import tornado.web

def install_client(ioloop, client, callback=None):
    def _tornado_handler(fd, events):
        if events | IOLoop.READ:
            client.handle_input()
        elif events | IOLoop.WRITE:
            client.handle_output()
        else:
            client.disconnect()
            if callback:
                callback(client)
    ioloop.add_handler(client.fd(), _tornado_handler, IOLoop.READ | IOLoop.WRITE | IOLoop.ERROR | IOLoop._EPOLLET)

def uninstall_client(ioloop, client):
    ioloop.remove_handler(client.fd())

class Camera(object):
    def __init__(self, client, name):
        self.name = name
        self._image_listeners = []
        self._location_listeners = []
        self._image = None
        self._client = client
        self._parameters = echocv.CameraIntrinsicsSubscriber(client, "%s.parameters" % name, lambda x: self._parameters_callback(x))
        self._location = echocv.CameraExtrinsicsSubscriber(client, "%s.location" % name, lambda x: self._location_callback(x))

    def _subscribe(self):
        self._image = echocv.ImageSubscriber(self._client, "%s.image" % self.name, lambda x: self._image_callback(x))

    def _unsubscribe(self):
        if len(self._image_listeners) == 0 and not self._image is None:
            self._image = None

    def listen_images(self, listener):
        self._image_listeners.append(listener)
        if len(self._image_listeners) == 1 and self._image is None:
            self._subscribe()

    def unlisten_images(self, listener):
        self._image_listeners.remove(listener)
        if len(self._image_listeners) == 0 and not self._image is None:
            self._unsubscribe()

    def listen_location(self, listener):
        self._location_listeners.append(listener)

    def unlisten_location(self, listener):
        self._location_listeners.remove(listener)

    def _distribute_image(self, image):       
        for c in self._image_listeners:
            c.push_image(image)
            
    def _distribute_location(self, location):       
        for c in self._location_listeners:
            c.push_camera_location(self, location)

    def _image_callback(self, image):
        if len(self._image_listeners) > 0:
            result, data = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if result == False:
                return
            self._distribute_image(str(data.data))

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
        
        self.write(self.boundary)
        self.write("\r\n")
        self.write("Content-type: image/jpeg\r\n")
        self.write("Content-length: %d\r\n\r\n" % len(image))
        self.write(image)
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
        self.write(image)
        self.finish()

    def on_finish(self):
        self.camera.unlisten_images(self)

    def on_connection_close(self):
        self.camera.unlisten_images(self)

    def check_etag_header(self):
        return False


__all__ = ["Camera", "ImageHandler", "VideoHandler", "install_client", "uninstall_client"]