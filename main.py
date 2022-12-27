from pixellib.instance import instance_segmentation

def object_detection_on_an_image():
    segment_image = instance_segmentation()
    segment_image.load_model('mask_rcnn_coco.h5')

    segment_image.segmentImage(
        image_path='1.jpg',
        output_image_name='output.jpg'
    )

def main():
    object_detection_on_an_image()


if __name__ == '__main__':
    main()