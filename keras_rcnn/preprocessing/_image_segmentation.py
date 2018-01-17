# -*- coding: utf-8 -*-

import keras.backend
import keras.preprocessing.image
import numpy
import skimage.io
import skimage.transform


def scale_size(size, min_size, max_size):
    """
    Rescales a given image size such that the larger axis is
    no larger than max_size and the smallest axis is as close
    as possible to min_size.
    """
    assert (len(size) == 2)

    scale = min_size / numpy.min(size)

    # Prevent the biggest axis from being larger than max_size.
    if numpy.round(scale * numpy.max(size)) > max_size:
        scale = max_size / numpy.max(size)

    rows, cols = size
    rows *= scale
    cols *= scale

    return (int(rows), int(cols)), scale


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(
            self,
            dictionary,
            classes,
            generator,
            target_shape=None,
            scale=1,
            ox=None,
            oy=None,
            batch_size=1,
            shuffle=False,
            seed=None
    ):
        self.dictionary = dictionary
        self.classes = classes
        self.generator = generator

        assert (len(self.dictionary) != 0)

        r = dictionary[0]["image"]["shape"]["r"]
        c = dictionary[0]["image"]["shape"]["c"]

        channels = dictionary[0]["image"]["shape"]["channels"]

        self.image_shape = (r, c, channels)
        self.scale = scale
        self.ox = ox
        self.oy = oy

        self.batch_size = batch_size

        if target_shape is None:
            self.target_shape, self.scale = scale_size(self.image_shape[0:2], numpy.min(self.image_shape[:2]), numpy.max(self.image_shape[:2]))

            self.target_shape = self.target_shape + (self.image_shape[2],)

        else:
            self.target_shape = target_shape + (self.image_shape[2],)

        # Metadata needs to be computed only once.
        r, c, channels = self.target_shape

        self.target_metadata = numpy.array([[r, c, self.scale]])

        super(DictionaryIterator, self).__init__(len(self.dictionary), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            selection = next(self.index_generator)

        return self._get_batches_of_transformed_samples(selection)

    def _get_batches_of_transformed_samples(self, selection):
        # Labels has num_classes + 1 elements, since 0 is reserved for
        # background.
        num_classes = len(self.classes)

        target_bounding_boxes = numpy.zeros((self.batch_size, 0, 4), dtype=keras.backend.floatx())

        target_images = numpy.zeros((self.batch_size,) + self.target_shape, dtype=keras.backend.floatx())

        target_masks = numpy.zeros((self.batch_size,) + self.target_shape, dtype=keras.backend.floatx())

        target_scores = numpy.zeros((self.batch_size, 0, num_classes + 1), dtype=numpy.uint8)

        for batch_index, image_index in enumerate(selection):
            count = 0

            while count == 0:
                # Image
                target_image_pathname = self.dictionary[image_index]["image"]["pathname"]

                target_image = skimage.io.imread(target_image_pathname)

                if target_image.ndim == 2:
                    target_image = numpy.expand_dims(target_image, -1)

                # crop
                if self.ox is None:
                    offset_x = numpy.random.randint(0, self.image_shape[1] - self.target_shape[1] + 1)
                else:
                    offset_x = self.ox

                if self.oy is None:
                    offset_y = numpy.random.randint(0, self.image_shape[0] - self.target_shape[0] + 1)
                else:
                    offset_y = self.oy

                target_image = target_image[offset_y:self.target_shape[0] + offset_y, offset_x:self.target_shape[1] + offset_x, :]

                # Copy image to batch blob.
                target_images[batch_index] = skimage.transform.rescale(target_image, scale=self.scale, mode="reflect")

                # Set ground truth boxes.
                for object_index, b in enumerate(self.dictionary[image_index]["objects"]):
                    if b["class"] not in self.classes:
                        continue

                    bounding_box = b["bounding_box"]

                    minimum_c = bounding_box["minimum"]["c"] - offset_x
                    minimum_r = bounding_box["minimum"]["r"] - offset_y

                    maximum_c = bounding_box["maximum"]["c"] - offset_x
                    maximum_r = bounding_box["maximum"]["r"] - offset_y

                    if maximum_c == target_image.shape[1]:
                        maximum_c -= 1

                    if maximum_r == target_image.shape[0]:
                        maximum_r -= 1

                    if minimum_c >= 0 and maximum_c < target_image.shape[1] and minimum_r >= 0 and maximum_r < target_image.shape[0]:
                        count += 1

                        target_bounding_box = [
                            minimum_c,
                            minimum_r,
                            maximum_c,
                            maximum_r
                        ]

                        target_bounding_boxes = numpy.append(target_bounding_boxes, [[target_bounding_box]], axis=1)

                        target_score = [0] * (num_classes + 1)

                        target_score[self.classes[b["class"]]] = 1

                        target_scores = numpy.append(target_scores, [[target_score]], axis=1)

            # Scale the ground truth boxes to the selected image scale.
            target_bounding_boxes[batch_index, :, :4] *= self.scale

        return [target_bounding_boxes, target_images, target_masks, self.target_metadata, target_scores], None


class ImageSegmentationGenerator:
    def flow(self, dictionary, classes, target_shape=None, scale=None, ox=None, oy=None, batch_size=1, shuffle=True, seed=None):
        return DictionaryIterator(dictionary, classes, self, target_shape, scale, ox, oy, batch_size, shuffle, seed)
