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
        // ...
        cvtColor(ct_slice, ct_slice, COLOR_HSV2BGR);
        
        // Normalisation
        normalize(ct_slice, ct_slice, 0, 255, NORM_MINMAX);
        
        // Égalisation de l'histogramme (si l'image est en niveaux de gris)
        cvtColor(ct_slice, ct_slice, COLOR_BGR2GRAY);
        equalizeHist(ct_slice, ct_slice);
        
        // Seuillage pour la segmentation
        Mat binary_image;
        threshold(ct_slice, binary_image, 128, 255, THRESH_BINARY);
        
        // Clonez l'image pour travailler avec une copie
        colour_ct_slice = ct_slice.clone();

        namedWindow("CT slice", WINDOW_AUTOSIZE);
        imshow("CT slice", colour_ct_slice);
        setMouseCallback("CT slice", &RegionGrowing::mouseCallback, this);
        waitKey(0);


        // Process each seed point separately
        for (size_t i = 0; i < seed_set.size(); ++i)
        {
            Mat segmented_image = regionGrowing(ct_slice, seed_set[i], 255, 2);

            // Display the segmented image for each seed point
            namedWindow("Segmentation", WINDOW_AUTOSIZE);
            imshow("Segmentation", segmented_image);
            waitKey(0);
        }

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
