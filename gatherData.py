import cv2

cascPath = '/Users/codruterdei/programming/openCVStuff/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

i = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CV_FEATURE_PARAMS_HAAR
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('cropped', cv2.getRectSubPix(frame, (x, y), (x+300, y+300)))
        cv2.imwrite('cropped.png', cv2.getRectSubPix(frame, (x, y), (w, h)))

        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = frame[ny:ny + nr, nx:nx + nr]
        lastimg = cv2.resize(faceimg, (128, 128))

        i += 1
        cv2.imwrite('tud{}.png'.format(i), lastimg)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or i == 500:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()





