# Imports
from my_package.model import InstanceSegmentationModel
from my_package.data import Dataset
from my_package.analysis import plot_visualization
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import numpy as np
from itertools import chain
from PIL import Image
import matplotlib.pyplot as plt
import os


def experiment(annotation_file, segmentor, transforms, outputs):
    '''
        Function to perform the desired experiments

        Arguments:
        annotation_file: Path to annotation file
        segmentor: The image segmentor
        transforms: List of transformation classes
        outputs: path of the output folder to store the images
    '''

    # Create the instance of the dataset
    dataset = Dataset(annotation_file, transforms)

    # Iterate over all data items
    for i in range(len(dataset)):
        # Get the data item
        data_item = dataset[i]

        # Get the image
        image = data_item['image']

        # Get the bounding boxes
        bboxes = data_item['gt_bboxes']

        # Get the predictions from the segmentor
        pred_boxes, pred_masks, pred_class, pred_score = segmentor(image)

        # Get the top 3 predictions
        if(len(pred_score) > 3):
            pred_boxes = pred_boxes[:3]
            pred_masks = pred_masks[:3]
            pred_class = pred_class[:3]
            pred_score = pred_score[:3]

        data_item['bboxes'] = []
        for j in range(len(pred_score)):
            temp_dict = {}
            temp_dict['bbox'] = list(chain.from_iterable(pred_boxes[j]))

            # Convert the bounding box to integers
            for k in range(len(temp_dict['bbox'])):
                temp_dict['bbox'][k] = int(temp_dict['bbox'][k])

            # Add the category and append to the dictionary
            temp_dict['category'] = pred_class[j]
            data_item['bboxes'].append(temp_dict)

        # Draw the segmentation maps on the image and save them
        image = plot_visualization(data_item, image.transpose((1, 2, 0)), str(i), 1)

    # Do the required analysis
    path = os.path.join(os.getcwd(), 'data').replace(os.sep, '/') + '/'
    my_image = Image.open(path + 'imgs/6.jpg')
    x, y = my_image.size

    # a) The original image
    dataset = Dataset(annotation_file, transforms=[])
    image_dict = dataset[6]
    plt.subplot(2, 4, 1)
    plt.title('Original image')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # b) Horizontally flipping
    dataset = Dataset(annotation_file, transforms=[FlipImage()])
    image_dict = dataset[6]
    plt.subplot(2, 4, 2)
    plt.title('Horizontally flipped image')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # c) Blurred image
    dataset = Dataset(annotation_file, transforms=[BlurImage(5)])
    image_dict = dataset[6]
    plt.subplot(2, 4, 3)
    plt.title('Blurred image (radius = 5)')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # d) Twice rescaled image
    dataset = Dataset(annotation_file, transforms=[RescaleImage((2 * y, 2 * x))])
    image_dict = dataset[6]
    plt.subplot(2, 4, 4)
    plt.title('2x rescaled image')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # e) Half rescaled image
    dataset = Dataset(annotation_file, transforms=[RescaleImage((y // 2, x // 2))])
    image_dict = dataset[6]
    plt.subplot(2, 4, 5)
    plt.title('0.5x rescaled image')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # f) 90 degrees right rotated image
    dataset = Dataset(annotation_file, transforms=[RotateImage(-90)])
    image_dict = dataset[6]
    plt.subplot(2, 4, 6)
    plt.title('90 degrees right rotated image')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # g) 45 degres left rotated image
    dataset = Dataset(annotation_file, transforms=[RotateImage(45)])
    image_dict = dataset[6]
    plt.subplot(2, 4, 7)
    plt.title('45 degrees left rotated image')
    plt.imshow(segmentedImage(image_dict, segmentor))

    # Show the graphs plotted and save the image
    plt.show()
    path = os.path.join(os.getcwd(), 'outputs').replace(os.sep, '/') + '/'
    plt.savefig(path + 'segmentation_visualization.png')


def segmentedImage(image_dict, segmentor):
    '''
        Function to predict the segmentation maps

        Arguments:
        image: The image to predict the segmentation maps
        image_dict: The dictionary containing the image information
        segmentor: The image segmentor
    '''

    image_dict['bboxes'] = []

    # Get the predictions from the segmentor
    image = image_dict['image']
    pred_boxes, pred_masks, pred_class, pred_score = segmentor(image)

    # Get the top 3 predictions
    if(len(pred_score) > 3):
        pred_boxes = pred_boxes[:3]
        pred_masks = pred_masks[:3]
        pred_class = pred_class[:3]
        pred_score = pred_score[:3]

    # Draw the segmentation maps on the image
    for j in range(len(pred_score)):
        temp_dict = {}
        temp_dict['bbox'] = list(chain.from_iterable(pred_boxes[j]))

        for k in range(len(temp_dict['bbox'])):
            temp_dict['bbox'][k] = int(temp_dict['bbox'][k])

        temp_dict['category'] = pred_class[j]
        image_dict['bboxes'].append(temp_dict)

    # Draw the segmentation maps on the image
    image = plot_visualization(image_dict, image.transpose((1, 2, 0)), None, 0)

    # Convert the image to numpy array
    segmentedImage = np.array(image)

    # Apply the mask to the image
    for mask in pred_masks:
        segmentedImage = segmentedImage + ((np.transpose(mask, (1, 2, 0))) * [0, 1, 0.5] * 255)

    # Convert the image numpy array to PIL image
    segmentedImage = Image.fromarray(np.uint8(segmentedImage)).convert('RGB')

    # Return the segmented image
    return segmentedImage


def main():
    # Initialize the segmentor
    segmentor = InstanceSegmentationModel()

    # Sample arguments to call experiment()
    experiment('./data/annotations.jsonl', segmentor, [], None)


if __name__ == '__main__':
    main()
