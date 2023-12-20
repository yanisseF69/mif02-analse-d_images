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

public:
    RegionGrowing(String image_path)
    {
        ct_slice = imread(image_path, IMREAD_COLOR);

        if (ct_slice.empty())
        {
            cerr << "Could not open or find the image" << endl;
            return; // No return -1; in a constructor
        }

        // Améliorations de l'image
        // Lissage
        GaussianBlur(ct_slice, ct_slice, Size(5, 5), 0);
        
        // Saturation (si l'image est en couleur)
        cvtColor(ct_slice, ct_slice, COLOR_BGR2HSV);
        // Ajuster la saturation dans le canal HSV si nécessaire

        ///////////////////////////////////////////////////////////////////////////////////////////// ça là c'est super important pour uniformiser les couleurs et rendre les regions plus lisibles pour notre code
            // Make colors more uniform (adjust saturation uniformly)
            float saturationFactor = 1.5;  // Adjust as needed

            // Split the image into channels
            vector<Mat> channels;
            split(ct_slice, channels);

            // Adjust the saturation channel (index 1 in the HSV color space)
            channels[1] = channels[1] * saturationFactor;

            // Merge the channels back into the image
            merge(channels, ct_slice);
        /////////////////////////////////////////////////////////////////////////////////////////////

        cvtColor(ct_slice, ct_slice, COLOR_HSV2BGR);
        
        // Normalisation
        normalize(ct_slice, ct_slice, 0, 255, NORM_MINMAX);
        
        // Égalisation de l'histogramme (si l'image est en niveaux de gris)
        cvtColor(ct_slice, ct_slice, COLOR_BGR2GRAY);
        equalizeHist(ct_slice, ct_slice);
        
        // Seuillage pour la segmentation
        Mat binary_image;
        threshold(ct_slice, binary_image, 0, 255, THRESH_BINARY);
        
        // Adaptive Thresholding
        adaptiveThreshold(ct_slice, binary_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

        // contours des objets - y a du potentiel on doit ameliorer ça        
        vector<vector<Point>> contours;
        findContours(binary_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Filter out small contours
        double minContourArea = 100.0;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > minContourArea) {
                // Process the contour
                drawContours(ct_slice, vector<vector<Point>>{contour}, 0, Scalar(0, 255, 0), 2);
            }
        }
        
        // Morphological Operations
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(binary_image, binary_image, MORPH_CLOSE, kernel);



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
