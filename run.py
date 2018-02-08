import keras_rcnn
import sklearn
import keras
import numpy
import matplotlib
import keras_rcnn.datasets.dsb2018
from sklearn.model_selection import train_test_split
from keras_rcnn.preprocessing import ObjectDetectionGenerator

from keras_rcnn.layers import RPN

training, test = keras_rcnn.datasets.dsb2018.load_data()

training, validation = sklearn.model_selection.train_test_split(training)

classes = {"rbc": 1, "not":2}

generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

generator = generator.flow(training, classes, (448, 448), 1.0)

validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

validation_data = validation_data.flow(validation, classes, (448, 448), 1.0)

# Create an instance of the RPN model:

image = keras.layers.Input((448, 448, 3))

model = RPN(image, classes=len(classes) + 1)

optimizer = keras.optimizers.Adam(0.0001)

model.compile(optimizer)

# Train the model:

model.fit_generator(
    epochs=10,
    generator=generator,
    steps_per_epoch=1000
)

# Predict and visualize your anchors or proposals:

example, _ = generator.next()

target_bounding_boxes, target_image, target_labels, _ = example

target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

target_image = numpy.squeeze(target_image)

target_labels = numpy.argmax(target_labels, -1)

target_labels = numpy.squeeze(target_labels)

output_anchors, output_proposals, output_deltas, output_scores = model.predict(example)

output_anchors = numpy.squeeze(output_anchors)

output_proposals = numpy.squeeze(output_proposals)

output_deltas = numpy.squeeze(output_deltas)

output_scores = numpy.squeeze(output_scores)

_, axis = matplotlib.pyplot.subplots(1)

axis.imshow(target_image)

for index, label in enumerate(target_labels):
    if label == 1:
        xy = [
            target_bounding_boxes[index][0],
            target_bounding_boxes[index][1]
        ]

        w = target_bounding_boxes[index][2] - target_bounding_boxes[index][0]
        h = target_bounding_boxes[index][3] - target_bounding_boxes[index][1]

        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="g", facecolor="none")

        axis.add_patch(rectangle)

for index, score in enumerate(output_scores):
    if score > 0.95:
        xy = [
            output_anchors[index][0],
            output_anchors[index][1]
        ]

        w = output_anchors[index][2] - output_anchors[index][0]
        h = output_anchors[index][3] - output_anchors[index][1]

        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

        axis.add_patch(rectangle)

matplotlib.pyplot.show()