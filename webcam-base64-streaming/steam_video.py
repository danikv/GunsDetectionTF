import asyncio
import websockets
import cv2
import base64


async def hello():
    cap = cv2.VideoCapture('friends.mp4')
    uri = "ws://localhost:3000"
    async with websockets.connect(uri) as websocket:
        while cap.isOpened():
            ret, frame = cap.read()
            retval, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = b'data:image/png;base64,' + base64.b64encode(buffer)
            await websocket.send(jpg_as_text)

asyncio.get_event_loop().run_until_complete(hello())