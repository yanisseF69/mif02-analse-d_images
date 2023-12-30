#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class RegionGrowing
{
private:
    vector<pair<int, int>> seed_set;
    Mat ct_slice;
    Mat colour_ct_slice;

    // TODO : faire du multi threading sur cette fonction car elle prend beaucoup de temps si on augmente k.
    void performKMeansSegmentation(const Mat& input_image, Mat& segmented_image) {
        int k = 15; // Nombre de clusters (vous pouvez ajuster cela)

        // Remodeler l'image en un tableau 2D de pixels
        Mat reshaped_image = input_image.reshape(1, input_image.rows * input_image.cols);

        // Convertir en type flottant pour K-means
        reshaped_image.convertTo(reshaped_image, CV_32F);

        // Clustering K-means
        Mat labels, centers;
        kmeans(reshaped_image, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_RANDOM_CENTERS, centers);

        // Remodeler le résultat à la taille originale de l'image
        Mat segmented_labels = labels.reshape(0, input_image.rows);

        // Convertir les étiquettes segmentées en 8 bits pour la visualisation
        segmented_labels.convertTo(segmented_image, CV_8U);

        // Normaliser l'image segmentée dans la plage [0, 255]
        normalize(segmented_image, segmented_image, 0, 255, NORM_MINMAX);
    }

    Mat regionSplittingAndMerging(const Mat& input_image, const Rect& region, float tolerance) {
        Mat output = Mat::zeros(input_image.size(), CV_8UC1);
        Mat region_of_interest = input_image(region);

        if (isHomogeneous(region_of_interest, tolerance)) {
            // If the region is homogeneous, fill it with the mean intensity
            output(region) = mean(region_of_interest)[0];
        } else {
            // If not, split the region into four quadrants and recursively process them
            Rect subregions[4];
            subregions[0] = Rect(region.x, region.y, region.width / 2, region.height / 2);
            subregions[1] = Rect(region.x + region.width / 2, region.y, region.width / 2, region.height / 2);
            subregions[2] = Rect(region.x, region.y + region.height / 2, region.width / 2, region.height / 2);
            subregions[3] = Rect(region.x + region.width / 2, region.y + region.height / 2, region.width / 2, region.height / 2);

            for (int i = 0; i < 4; ++i) {
                Mat subregion_mask = Mat::zeros(input_image.size(), CV_8UC1);
                rectangle(subregion_mask, subregions[i], Scalar(255), FILLED);

                Mat subregion_output = regionSplittingAndMerging(input_image, subregions[i], tolerance);

                bitwise_or(output, subregion_output, output, subregion_mask);
            }
        }

        return output;
    }

    bool isHomogeneous(const Mat& region, float tolerance) {
        Scalar mean_val = mean(region);
        Mat diff;
        absdiff(region, mean_val, diff);
        Scalar diff_mean = mean(diff);

        return (diff_mean[0] <= tolerance);
    }


public:
    RegionGrowing(String image_path)
    {
        ct_slice = imread(image_path, IMREAD_GRAYSCALE);

        if (ct_slice.empty())
        {
            cerr << "Could not open or find the image" << endl;
            return; // No return -1; in a constructor
        }

        performKMeansSegmentation(ct_slice, ct_slice);

        // Clonez l'image pour travailler avec une copie
        colour_ct_slice = ct_slice.clone();

        namedWindow("CT slice", WINDOW_AUTOSIZE);
        imshow("CT slice", colour_ct_slice);

        // Automatically generate seed points at regular intervals
        int interval = 60; // Adjust the interval based on your preference
        for (int y = 0; y < ct_slice.rows; y += interval) {
            for (int x = 0; x < ct_slice.cols; x += interval) {
                seed_set.push_back(pair<int, int>(x, y));
            }
        }

        // Display seed points
        for (const auto& seed : seed_set) {
            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            circle(colour_ct_slice, Point(seed.first, seed.second), 4, color, FILLED);
        }
        imshow("CT slice", colour_ct_slice);
        waitKey(0);

        // Process each seed point separately
        vector<Mat> segmented_regions;
        for (size_t i = 0; i < seed_set.size(); ++i)
        {
            Mat segmented_image = regionGrowing(ct_slice, seed_set[i], 255, 2);
            segmented_regions.push_back(segmented_image);
        }

        // Display all segmented regions in one window with different colors
        Mat all_regions = Mat::zeros(ct_slice.size(), CV_8UC3);
        for (size_t i = 0; i < seed_set.size(); ++i)
        {
            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            bitwise_or(all_regions, Scalar(0, 0, 0), all_regions, segmented_regions[i] == 255);
            bitwise_or(all_regions, color, all_regions, segmented_regions[i] == 255);
        }

        namedWindow("All Segmented Regions", WINDOW_AUTOSIZE);
        imshow("All Segmented Regions", all_regions);
        waitKey(0);

        // Apply region splitting and merging on the whole image
        Mat rsm_result = regionSplittingAndMerging(ct_slice, Rect(0, 0, ct_slice.cols, ct_slice.rows), 10);

        namedWindow("Region Splitting and Merging Result", WINDOW_AUTOSIZE);
        imshow("Region Splitting and Merging Result", rsm_result);
        waitKey(0);

        destroyAllWindows();
    }

    ~RegionGrowing()
    {
    }

    static void mouseCallback(int event, int x, int y, int flags, void *userdata)
    {
        RegionGrowing *instance = static_cast<RegionGrowing *>(userdata);

        if (event == EVENT_LBUTTONDOWN)
        {
            instance->seed_set.push_back(pair<int, int>(x, y));

            // Generate a unique color for each seed point
            Scalar color(rand() & 255, rand() & 255, rand() & 255);
            circle(instance->colour_ct_slice, Point(x, y), 4, color, FILLED);
            imshow("CT slice", instance->colour_ct_slice);
        }
    }

    Mat regionGrowing(const Mat &anImage,
                      const pair<int, int> &seedPoint,
                      unsigned char anInValue = 255,
                      float tolerance = 5)
    {
        Mat visited_matrix = Mat::zeros(Size(anImage.cols, anImage.rows), CV_8UC1);
        vector<pair<int, int>> point_list;
        point_list.push_back(seedPoint);

        while (!point_list.empty())
        {
            pair<int, int> this_point = point_list.back();
            point_list.pop_back();

            int x = this_point.first;
            int y = this_point.second;
            unsigned char pixel_value = anImage.at<unsigned char>(Point(x, y));

            visited_matrix.at<unsigned char>(Point(x, y)) = anInValue;

            for (int j = y - 1; j <= y + 1; ++j)
            {
                if (0 <= j && j < anImage.rows)
                {
                    for (int i = x - 1; i <= x + 1; ++i)
                    {
                        if (0 <= i && i < anImage.cols)
                        {
                            unsigned char neighbour_value = anImage.at<unsigned char>(Point(i, j));
                            unsigned char neighbour_visited = visited_matrix.at<unsigned char>(Point(i, j));

                            if (!neighbour_visited &&
                                fabs(neighbour_value - pixel_value) <= (tolerance / 100.0 * 255.0))
                            {
                                point_list.push_back(pair<int, int>(i, j));
                                visited_matrix.at<unsigned char>(Point(i, j)) = anInValue;
                            }
                        }
                    }
                }
            }
        }

        return visited_matrix;
    }
};

int main(int argc, char **argv)
{
    RegionGrowing rg("data/000451.jpg");
    return 0;
}
