import keras
import numpy
import matplotlib
import keras_rcnn.datasets.dsb2018
import sklearn.model_selection
import keras_rcnn.preprocessing

from keras_rcnn.models import RPN

training, test = keras_rcnn.datasets.dsb2018.load_data()

training, validation = sklearn.model_selection.train_test_split(training)

classes = {"nucleus": 1}

generator = keras_rcnn.preprocessing.ImageSegmentationGenerator()

generator = generator.flow(training, classes, (256, 256), 1.0, ox=0, oy=0)

validation_data = keras_rcnn.preprocessing.ImageSegmentationGenerator()

validation_data = validation_data.flow(validation, classes, (256, 256), 1.0)


(target_bounding_boxes, target_image, target_masks, _, target_scores), _ = generator.next()
target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

target_image = numpy.squeeze(target_image)

target_scores = numpy.argmax(target_scores, -1)

target_scores = numpy.squeeze(target_scores)

_, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

axis.imshow(target_image)

for target_index, target_score in enumerate(target_scores):
    if target_score > 0:
        xy = [
            target_bounding_boxes[target_index][0],
            target_bounding_boxes[target_index][1]
        ]

        w = target_bounding_boxes[target_index][2] - target_bounding_boxes[target_index][0]
        h = target_bounding_boxes[target_index][3] - target_bounding_boxes[target_index][1]

        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

        axis.add_patch(rectangle)

matplotlib.pyplot.show()





# Create an instance of the RPN model:

image = keras.layers.Input((256, 256, 3))

model = RPN(image, classes=len(classes) + 1, feature_maps=[32, 64, 128, 256])

optimizer = keras.optimizers.Adam(0.0001)

model.compile(optimizer)

# Train the model:

model.fit_generator(
    epochs=10,
    generator=generator,
    steps_per_epoch=1000
)
#
# # Predict and visualize your anchors or proposals:
#
# example, _ = generator.next()
#
# target_bounding_boxes, target_image, target_labels, _ = example
#
# target_bounding_boxes = numpy.squeeze(target_bounding_boxes)
#
# target_image = numpy.squeeze(target_image)
#
# target_labels = numpy.argmax(target_labels, -1)
#
# target_labels = numpy.squeeze(target_labels)
#
# output_anchors, output_proposals, output_deltas, output_scores = model.predict(example)
#
# output_anchors = numpy.squeeze(output_anchors)
#
# output_proposals = numpy.squeeze(output_proposals)
#
# output_deltas = numpy.squeeze(output_deltas)
#
# output_scores = numpy.squeeze(output_scores)
#
# _, axis = matplotlib.pyplot.subplots(1)
#
# axis.imshow(target_image)
#
# for index, label in enumerate(target_labels):
#     if label == 1:
#         xy = [
#             target_bounding_boxes[index][0],
#             target_bounding_boxes[index][1]
#         ]
#
#         w = target_bounding_boxes[index][2] - target_bounding_boxes[index][0]
#         h = target_bounding_boxes[index][3] - target_bounding_boxes[index][1]
#
#         rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="g", facecolor="none")
#
#         axis.add_patch(rectangle)
#
# for index, score in enumerate(output_scores):
#     if score > 0.95:
#         xy = [
#             output_anchors[index][0],
#             output_anchors[index][1]
#         ]
#
#         w = output_anchors[index][2] - output_anchors[index][0]
#         h = output_anchors[index][3] - output_anchors[index][1]
#
#         rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")
#
#         axis.add_patch(rectangle)
#
# matplotlib.pyplot.show()