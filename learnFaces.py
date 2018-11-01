import tensorflow as tf
from tensorflow import keras

import numpy as np
import cv2


cascPath = '/Users/codruterdei/programming/openCVStuff/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)



def load_data():

    # preparing the as in
    # getting the images from the disk and storing them into memory
    # converting them to grayscale
    # resizing them to 32x32 pixels so the nn won't take too long to train
    cod = [cv2.resize(cv2.cvtColor(cv2.imread('codFace3/tud{}.png'.format(x)), cv2.COLOR_BGR2GRAY), dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for x in range(1, 61)]#101)]
    boi = [cv2.resize(cv2.cvtColor(cv2.imread('boiFace3/tud{}.png'.format(x)), cv2.COLOR_BGR2GRAY), dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for x in range(1, 61)]#101)]
    cat = [cv2.resize(cv2.cvtColor(cv2.imread('catFace3/tud{}.png'.format(x)), cv2.COLOR_BGR2GRAY), dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for x in range(1, 61)]#101)]
    vik = [cv2.resize(cv2.cvtColor(cv2.imread('vikFace3/tud{}.png'.format(x)), cv2.COLOR_BGR2GRAY), dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for x in range(1, 61)]#101)]

    # concatenating all the lists of images
    [cod.append(x) for x in boi]
    [cod.append(x) for x in cat]
    [cod.append(x) for x in vik]

    # generating labels for every picture
    # labels are just the name of the thing you wanna classify (1) <- more on this later
    lablesCod = [0 for _ in range(1, 61)]
    lablesBoi = [1 for _ in range(1, 61)]
    lablesCat = [2 for _ in range(1, 61)]
    lablesVik = [3 for _ in range(1, 61)]

    [lablesCod.append(x) for x in lablesBoi]
    [lablesCod.append(x) for x in lablesCat]
    [lablesCod.append(x) for x in lablesVik]

    # returning the stuff i need in a format that the library accepts it
    return np.array([cv2.Canny(x, 100, 200) for x in cod]), lablesCod


train_images, train_labels = load_data()

# (1) i've given the labels 0 1 2 3 and now i'm giving them a name
# labeling is just giving an identifier to the data you've given the algorithm so it knows what to tell you the
# when it's giving you an answer
class_names = ['cod', 'boi', 'cat', 'vik']

# normalizing the data = all the images are stored in a pixel matrix
# every element being a number from 0 to 255, and i want instead a number between 0 and 1
train_images = train_images / 255.0


# preparing the model and modeling it
# this is where you configure the network
# you give mathematical functions for activation and you tell em hoy many neurons are per layer
# the first case is 128 neurons aka 2^7
# dense means that are fully connected, so 2 layers make a fully connected graph
model = tf.keras.models.Sequential([
    # input layer where i give it a 1d array, meaning the flatten images.
    # flatten is the images in a straight line rather than a matrix
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(2**7, activation=tf.nn.relu),
    keras.layers.Dense(2**10, activation=tf.nn.softsign),
    keras.layers.Dense(2**10, activation=tf.nn.sigmoid),
    # here is the ouput layer and want to classify between 4 things, so i'm giving
    # it 4 neurons that will return 4 numbers that will add up to 1
    # that means that i get the probability for every choice

    keras.layers.Dense(4, activation=tf.nn.softmax)
])

# compiling the model. it generates alll the neurons and now it waits for me to train it
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training is where i give it all the data and the labels so it can learn
# epochs are the number of iterations that the model will go trough in the learning phase
model.fit(train_images, train_labels, epochs=70)


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to H63DF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def predict():
    # preparing the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # frame will be the webcam image each frame
        ret, frame = cap.read()

        # converting the image from the webcam to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # this will give me an array of faces or rather the coordinates of faces it finds
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CV_FEATURE_PARAMS_HAAR
        )

        # Iterating through the faces in order to classify them
        for (x, y, w, h) in faces:

            # umplutura
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            # the actual pixels of a face
            faceimg = frame[ny:ny + nr, nx:nx + nr]
            # resizing the picture in order for it to be fed to the model
            lastimg = cv2.resize(faceimg, (32, 32))
            # grayscaling it
            lastimg = cv2.Canny(cv2.cvtColor(lastimg, cv2.COLOR_BGR2GRAY), 100, 200)

            # you have to make an array of the images in order to feed it to the model even tho you give it one
            # this translates to
            # the model is really excited to give you many answers it doesn't want just one question
            img = (np.expand_dims(lastimg, 0))

            # here i'm getting an array of predictions in the form of the previous mentioned
            # a list of probabilities
            prediction = model.predict(img)

            # here i'm deciding which color the rectangle will be depending on the answer the model gives me
            # to note is that np.argmax tells me on which position in the list we find the highest number
            color = (255, 0, 0)
            if np.argmax(prediction) == 0:
                color = (0, 255, 0)
            if np.argmax(prediction) == 1:
                color = (0, 0, 0)
            if np.argmax(prediction) == 2:
                color = (0, 255, 255)
            if np.argmax(prediction) == 3:
                color = (280, 120, 280)

            # drawing the rectangle on the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # and finally showing the image with the rectangles drawn on it
        cv2.imshow('ics', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            cap.release()
            cv2.destroyAllWindows()

predict()
