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

    void performKMeansSegmentation(const Mat& input_image, Mat& segmented_image) {
        int k = 30; // Nombre de clusters (vous pouvez ajuster cela)

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

public:
    RegionGrowing(String image_path)
    {
        ct_slice = imread(image_path, IMREAD_COLOR);

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
        setMouseCallback("CT slice", &RegionGrowing::mouseCallback, this);
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
