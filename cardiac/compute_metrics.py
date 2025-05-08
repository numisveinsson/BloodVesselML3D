import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import glob


def hausdorff(pred, truth):

    # check pixel type
    if truth.GetPixelID() != sitk.sitkUInt8:
        truth = sitk.Cast(truth, sitk.sitkUInt8)
    if pred.GetPixelID() != sitk.sitkUInt8:
        pred = sitk.Cast(pred, sitk.sitkUInt8)

    # check that image origin is the same
    if truth.GetOrigin() != pred.GetOrigin():
        pred.SetOrigin(truth.GetOrigin())

    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.Execute(pred, truth)

    # Get 95th percentile Hausdorff distance

    return haus_filter.GetAverageHausdorffDistance()


def hausdorf_95(pred, truth):
    from SimpleITK import GetArrayViewFromImage as ArrayView
    from functools import partial

    prediction = pred
    gold = truth
    perc = 100

    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)

    num_labels = 1
    for label in range(1, num_labels + 1):
        gold_surface = sitk.LabelContour(gold == label, False)
        prediction_surface = sitk.LabelContour(prediction == label, False)

        # Get distance map for contours (the distance map computes the minimum distances)
        prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
        gold_distance_map = sitk.Abs(distance_map(gold_surface))

        # Find the distances to surface points of the contour.  Calculate in both directions
        gold_to_prediction = ArrayView(prediction_distance_map)[ArrayView(gold_surface) == 1]
        prediction_to_gold = ArrayView(gold_distance_map)[ArrayView(prediction_surface) == 1]

        # Find the 95% Distance for each direction and average
        # print((np.percentile(prediction_to_gold, perc) + np.percentile(gold_to_prediction, perc)) / 2.0)

        return (np.percentile(prediction_to_gold, perc) + np.percentile(gold_to_prediction, perc)) / 2.0


if __name__ == '__main__':
    dir_gt = "/Users/numisveins/Documents/data_papers/data_combo_paper/ct_data/gt_cardiac_segs/"
    dir_pred = "/Users/numisveins/Documents/data_papers/data_combo_paper/ct_data/meshes/transformed_to_seg_vti/new_format/"
    dir_out = "/Users/numisveins/Documents/data_papers/data_combo_paper/ct_data/hausdorff_distances/"

    extension = ".mha"
    gt_files = glob.glob(os.path.join(dir_gt, "*" + extension))
    filenames = [os.path.basename(gt) for gt in gt_files]
    pred_files = [os.path.join(dir_pred, filename) for filename in filenames]
    # sort
    gt_files = sorted(gt_files)
    pred_files = sorted(pred_files)
    # remove last two
    gt_files = gt_files[:-2]
    pred_files = pred_files[:-2]
    # assert all(os.path.isfile(f) for f in pred_files)
    hausdorff_distances = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        print("Processing : ", os.path.basename(gt_file))
        gt_mesh = sitk.ReadImage(gt_file)
        pred_mesh = sitk.ReadImage(pred_file)

        gt_classes = np.unique(sitk.GetArrayFromImage(gt_mesh))[1:]
        pred_classes = np.unique(sitk.GetArrayFromImage(pred_mesh))[1:-1]  # remove the last class which is the background
        # make integer
        gt_classes = gt_classes.astype(int)
        pred_classes = pred_classes.astype(int)
        import pdb
        # pdb.set_trace()
        assert len(gt_classes) == len(pred_classes), "Number of classes in gt and pred do not match"

        # Compute Hausdorff distance for each class
        for gt_class, pred_class in zip(gt_classes, pred_classes):
            gt_class = int(gt_class)
            pred_class = int(pred_class)
            gt_mask = sitk.BinaryThreshold(gt_mesh, lowerThreshold=gt_class, upperThreshold=gt_class)
            pred_mask = sitk.BinaryThreshold(pred_mesh, lowerThreshold=pred_class, upperThreshold=pred_class)

            # Compute Hausdorff distance
            hausdorff_distance = hausdorf_95(pred_mask, gt_mask)

            print(f"Gt class: {gt_class}, Pred class: {pred_class}, Hausdorff distance: {hausdorff_distance}")

            hausdorff_distances.append(hausdorff_distance)

    # # Print the distances
    # for i, filename in enumerate(filenames):
    #     print(f"{filename}: {hausdorff_distances[i]}")

    # # Create output directory if it doesn't exist
    # os.makedirs(dir_out, exist_ok=True)

    # # Save the distances to a CSV file
    # data = pd.DataFrame(hausdorff_distances, index=filenames)
    # output_file_path = os.path.join(dir_out, "hausdorff_distance.csv")
    # data.to_csv(output_file_path)