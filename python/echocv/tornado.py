
from __future__ import absolute_import

import math
from asyncio import Future, get_event_loop

from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
import tornado.web
import cv2

import echolib
import echocv

class TornadoClientObserver(echolib.IOBaseObserver):

    def __init__(self, ioloop, client, disconnect_callback):
        super().__init__()
        self._ioloop = ioloop
        self._client = client
        self._canwrite = False
        self._disconnect_callback = disconnect_callback
        self._client.observe(self)

        if isinstance(ioloop, IOLoop):
            ioloop.add_handler(client.fd(), self._ioevent, IOLoop.READ | IOLoop.ERROR)


    def on_output(self, _):
        if self._canwrite:
            return

        self._ioloop.update_handler(self._client.fd(), IOLoop.READ | IOLoop.ERROR | IOLoop.WRITE)

    def _ioevent(self, _, events):
        if events & IOLoop.ERROR:
            self._client.disconnect()
            if self._disconnect_callback:
                self._self_disconnect_callback(self._client)
        else:
            if events & IOLoop.READ:
                self._client.handle_input()
            if events & IOLoop.WRITE:
                if self._client.handle_output():
                    self._ioloop.update_handler(self._client.fd(), IOLoop.READ | IOLoop.ERROR)
                    self._canwrite = False


def install_client(ioloop, client, disconnect_callback=None):
    TornadoClientObserver(ioloop, client, disconnect_callback)

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
            self._jpeg = bytes(data.data)
        return self._jpeg

class FutureCallback(object):

    def __init__(self):
        self._future = Future()

    def __call__(self, _, result):
        self._future.set_result(result)

    async def wait(self):
        return await self._future

class Camera(object):
    def __init__(self, client, name):
        self.name = name
        self._image_listeners = []
        self._location_listeners = []
        self._client = client
        self._parameters = None
        self._parameters_sub = echocv.CameraIntrinsicsSubscriber(client, "%s.parameters" % name, self._parameters_callback)
        self._location_sub = None
        self._image_sub = None

    async def image(self):
        cb = FutureCallback()
        self.listen_images(cb)
        result = await cb.wait()
        self.unlisten_images(cb)
        return result

    async def location(self):
        cb = FutureCallback()
        self.listen_location(cb)
        result = await cb.wait()
        self.unlisten_location(cb)
        return result

    def listen_images(self, listener):
        self._image_listeners.append(listener)
        if len(self._image_listeners) == 1 and self._image_sub is None:
            self._image_sub = echocv.FrameSubscriber(self._client, "%s.image" % self.name,self._frame_callback)

    def unlisten_images(self, listener):
        self._image_listeners.remove(listener)
        if len(self._image_listeners) == 0 and not self._image_sub is None:
            self._image_sub = None

    def listen_location(self, listener):
        self._location_listeners.append(listener)
        if len(self._location_listeners) == 1 and self._location_sub is None:
            self._location_sub = echocv.CameraExtrinsicsSubscriber(self._client, "%s.location" % self.name, self._location_callback)

    def unlisten_location(self, listener):
        self._location_listeners.remove(listener)
        if len(self._location_listeners) == 0 and not self._location_sub is None:
            self._location_sub = None

    def _distribute_image(self, image):
        for c in self._image_listeners:
            c(self, image)
            
    def _distribute_location(self, location):
        for c in self._location_listeners:
            c(self, location)

    def _frame_callback(self, frame):
        if len(self._image_listeners) > 0:
            img = Image(frame)
            self._distribute_image(img)

    def _location_callback(self, location):
        self._distribute_location(location)

    def _parameters_callback(self, parameters):
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

class VideoHandler(tornado.web.RequestHandler):

    def __init__(self, application, request, camera):
        super(VideoHandler, self).__init__(application, request)
        self.camera = camera
        self.boundary = '--imageboundary--'
        self._future = None
        self._running = True

    async def get(self):
        # TODO: add random chars to boundary

        self.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=' + self.boundary)
        self.flush()


        self._future = Future()
        self.camera.listen_images(self._image_cb)

        try:
            while self._running:
                image = await self._future
                self._future = Future()
                if len(self.boundary) < 1:
                    return
                
                data = image.jpeg()

                self.write(self.boundary)
                self.write("\r\n")
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %d\r\n\r\n" % len(data))
                self.write(data)
                self.write("\r\n")
                await self.flush()
        except StreamClosedError:
            pass

        self.camera.unlisten_images(self._image_cb)

    def _image_cb(self, _, image):
        if not self._future.done():
            self._future.set_result(image)

    def on_finish(self):
        self._running = False

    def on_connection_close(self):
        self._running = False

    def check_etag_header(self):
        return False

class ImageHandler(tornado.web.RequestHandler):
    def __init__(self, application, request, camera):
        super(ImageHandler, self).__init__(application, request)
        self.camera = camera

    def set_default_headers(self):
        self.set_header('Content-Type', 'image/jpeg')
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')

    async def get(self):
        image = await self.camera.image()
        self.set_header('X-Timestamp', image.timestamp().isoformat())
        self.write(image.jpeg())
        self.finish()

    def check_etag_header(self):
        return False


__all__ = ["Camera", "ImageHandler", "VideoHandler", "install_client", "uninstall_client"]