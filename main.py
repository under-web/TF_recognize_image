import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pixellib.instance import instance_segmentation

def object_detection_on_an_image():
    segment_image = instance_segmentation()
    segment_image.load_model('mask_rcnn_coco.h5')

    target_class = segment_image.select_target_classes(person=True)

    result = segment_image.segmentImage(
        image_path='1.jpg',
        show_bboxes=True,
        segment_target_classes=target_class,
        extract_segmented_objects=True,
        save_extracted_objects=True,
        output_image_name='output.jpg'
    )
    print(result[0]["scores"])
    obj_count = len(result[0]["scores"])

    print(f"{obj_count} человек на фото")
def main():
    object_detection_on_an_image()


if __name__ == '__main__':
    main()