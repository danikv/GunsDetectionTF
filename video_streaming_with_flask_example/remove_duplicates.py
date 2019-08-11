import cv2
import constants


def remove_duplicates(gun_files_queue, ref_images):
    orb = cv2.ORB_create()

    ref_descriptors = get_ref_models_descriptors(orb, ref_images)

    saved_images = []
    current_saved_img_id = 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        current_image = gun_files_queue.get()
        current_image = cv2.resize(current_image, (300, 300), interpolation=cv2.INTER_LINEAR)

        kp1, des1 = orb.detectAndCompute(current_image, None)

        # no features found
        if des1 is None:
            continue

        name = 'guns_{}'.format(current_saved_img_id)

        #pass only images that are close enough to reference images
        ref_matches = []
        for des in ref_descriptors:
            ref_matches.append(bf.match(des1, des))

        if is_image_far_from_refs(ref_matches, constants.REFERENCE_MAX_DIST):
            continue

        # add first image
        if not saved_images:
            saved_images.append(current_image)
            cv2.imwrite('static/images/guns/{}.jpg'.format(name), current_image)
            current_saved_img_id += 1
            continue

        should_update = True

        for saved_image in saved_images:
            # find the keypoints and descriptors with ORB
            kp2, des2 = orb.detectAndCompute(saved_image, None)
            matches = bf.match(des1, des2)

            if not matches:
                continue

            if not is_image_far_from_refs([matches], constants.SIMILARITY_MAX_DIST):
                print('daniel2')
                should_update = False
                break

        if should_update:
            print('daniel3')
            saved_images.append(current_image)
            cv2.imwrite('static/images/guns/{}.jpg'.format(name), current_image)
            current_saved_img_id += 1


def is_image_far_from_refs(ref_matches, max_dist):
    is_far = True
    for ref_match in ref_matches:
        filtered = [x.distance for x in ref_match if x.distance < constants.OUTLIERS_MIN_DIST]
        if not filtered:
            continue

        avg = sum(filtered) / len(filtered)
        if avg <= max_dist:
            is_far = False

    return is_far

def get_ref_models_descriptors(orb, ref_models):
    ref_descriptors = []
    for reference in ref_models:
        ref_image = cv2.imread(reference)
        _, des_ref = orb.detectAndCompute(ref_image, None)
        ref_descriptors.append(des_ref)
    return ref_descriptors
