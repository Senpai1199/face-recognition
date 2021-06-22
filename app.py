from flask import Flask, render_template, Response, request, redirect, url_for
from imutils.video import VideoStream
from pyimagesearch.motion_detection import SingleMotionDetector
import threading
import argparse
import datetime
import imutils
import time
import cv2
import re  # regex matching for email
import json
import time
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # grab the current timestamp and draw it on the frame
        # timestamp = datetime.datetime.now()
        # cv2.putText(frame, timestamp.strftime(
        #     "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = md.detect(gray)

            # cehck to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == "GET":
        return render_template('signup.html')

    if request.method == "POST":
        # print("*** SIGNUP REQUEST RECEIVED **")
        try:
            email = str(request.form['email'])
            password = str(request.form['password1'])
            confirm_password = str(request.form['password2'])
            # print(email)
            # print(password)
            # print(confirm_password)

            if password != confirm_password:
                # print('Passwords dont match')
                return redirect(url_for('signup'))  # signup fail

            # # compute embeddings from image clicked by user
            # img = 'dsad.jpg'
            with open('db.json', 'r') as f:
                dbDataDict = json.loads(f.read())
            dbDataDict[email] = password

            with open('db.json', 'w') as f:
                json.dump(dbDataDict, f)
            # dbDataJson = json.dumps(dbData)
            # print("Db data json: ", dbDataJson)
            # with open("db.json", "w") as outfile:
            #     print("Inside with open")
            #     json.dump(dbDataJson, outfile)
            f.close()
            # outfile.close()
            # print("SignUp Successful!")
            time.sleep(1)
            return redirect(url_for('index'))
        except:
            return redirect(url_for('signup'))  # signup fail


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == "GET":
        return render_template('login.html')

    if request.method == "POST":
        # print("*** LOGIN REQUEST RECEIVED **")
        try:
            email = str(request.form['email'])
            password = str(request.form['password'])
            with open('db.json', 'r') as f:
                dbDataDict = json.loads(f.read())
            if email in dbDataDict.keys():
                print("User exists")
                if password == dbDataDict[email]:
                    time.sleep(1)
                    return redirect(url_for('home'))
                else:
                    print("Login fail")
                    return redirect(url_for('index'))  # login fail
            else:
                print("Login fail")
                return redirect(url_for('index'))  # login fail
        except:
            print("Login fail")
            return redirect(url_for('index'))  # login fail


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()

# app.run(Debug=True)
