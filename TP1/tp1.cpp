#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace std;
using namespace cv;

class RegionGrowing
{
    private:
        vector<pair<int, int>> seed_set;
        Mat ct_slice;
        Mat colour_ct_slice;
        RNG rng;

        /** 
         * Performs K-Means segmentation on the input image.
         * 
         * @param input_image The input image to be segmented.
         * @param segmented_image The output segmented image.
         */
        void performKMeansSegmentation(const Mat& input_image, Mat& segmented_image) {
            // TODO : faire du multi threading sur cette fonction car elle prend beaucoup de temps si on augmente k.
            int k = 10; // Nombre de clusters (ajustable)

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

        /**
         * Performs region splitting and merging on the input color image.
         * 
         * @param input_image The input color image to be processed.
         * @param region The region of interest within the image.
         * @param tolerance The tolerance value for checking region homogeneity.
         * @return The processed color image after region splitting and merging.
         */
        void processSubregion(const Mat& input_image, const Rect& region, float tolerance, Mat& output, RNG& rng) {
            Mat region_of_interest = input_image(region);

            if (isHomogeneousColor(region_of_interest, tolerance)) {
                // Si la région est homogène, la remplir avec une couleur aléatoire
                output(region) = mean(region_of_interest);
            } else {
                // Sinon, diviser la région en quatre quadrants et les traiter de manière récursive
                Rect subregions[4];
                subregions[0] = Rect(region.x, region.y, region.width / 2, region.height / 2);
                subregions[1] = Rect(region.x + region.width / 2, region.y, region.width / 2, region.height / 2);
                subregions[2] = Rect(region.x, region.y + region.height / 2, region.width / 2, region.height / 2);
                subregions[3] = Rect(region.x + region.width / 2, region.y + region.height / 2, region.width / 2, region.height / 2);

                // Générer une couleur aléatoire pour ce niveau de segmentation
                Scalar random_color(rand() % 256, rand() % 256, rand() % 256);

                std::vector<std::thread> threads;

                for (int i = 0; i < 4; ++i) {
                    threads.emplace_back([this, &input_image, &subregions, i, &tolerance, &output, &rng, random_color] {
                        Mat subregion_mask = Mat::zeros(input_image.size(), CV_8UC1);
                        rectangle(subregion_mask, subregions[i], Scalar(255), FILLED);

                        cv::Mat subregion_output = cv::Mat::zeros(input_image.size(), CV_8UC1);

                        processSubregion(input_image, subregions[i], tolerance, subregion_output, rng);

                        bitwise_or(output, subregion_output, output);
                    });
                }

                // Attendre que tous les threads aient terminé
                for (auto& thread : threads) {
                    thread.join();
                }
            }
        }

        Mat regionSplittingAndMerging(const Mat& input_image, const Rect& region, float tolerance, RNG rng) {
            Mat output = Mat::zeros(input_image.size(), CV_8UC1);
            processSubregion(input_image, region, tolerance, output, rng);
            return output;
        }

        /**
         * Checks if a color region is homogeneous based on the mean intensity and tolerance.
         * 
         * @param region The color region to be checked for homogeneity.
         * @param tolerance The tolerance value for determining homogeneity.
         * @return True if the region is homogeneous, false otherwise.
         */
        bool isHomogeneousColor(const Mat& region, float tolerance) {
            Scalar mean_val = mean(region);
            Mat diff;
            absdiff(region, mean_val, diff);
            Scalar diff_mean = mean(diff);

            return (diff_mean[0] <= tolerance && diff_mean[1] <= tolerance && diff_mean[2] <= tolerance);
        }

        /**
         * Generates a specified number of random seed points within the image dimensions.
         * 
         * @param numPoints The number of random seed points to generate.
         */
        void generateRandomSeedPoints(int numPoints) {
            for (int i = 0; i < numPoints; ++i) {
                int x = rand() % ct_slice.cols;
                int y = rand() % ct_slice.rows;
                seed_set.push_back(pair<int, int>(x, y));
            }
        }




    public:

        /**
         * Constructor for the RegionGrowing class.
         * 
         * Initializes the instance with an input color image, generates seed points,
         * performs region growing segmentation, and displays the results.
         * 
         * @param image_path Path to the input color image.
         */
        RegionGrowing(String image_path)
        {
            rng = RNG(12345);
            ct_slice = imread(image_path, IMREAD_COLOR);

            if (ct_slice.empty())
            {
                cerr << "Could not open or find the image" << endl;
                return; // No return -1; in a constructor
            }

            //performKMeansSegmentation(ct_slice, ct_slice);
            Mat grayImage;
            cvtColor(ct_slice, grayImage, COLOR_BGR2GRAY);

            // Clonez l'image pour travailler avec une copie
            colour_ct_slice = ct_slice.clone();

            namedWindow("CT slice", WINDOW_AUTOSIZE);
            // imshow("CT slice", colour_ct_slice);
            // waitKey(0);

            /* ------ generation des points tous les n pixels -------- /!\ pas optimal 
                    // Automatically generate seed points at regular intervals
                    int interval = 60; // Adjust the interval based on your preference
                    for (int y = 0; y < ct_slice.rows; y += interval) {
                        for (int x = 0; x < ct_slice.cols; x += interval) {
                            seed_set.push_back(pair<int, int>(x, y));
                        }
                    }
            */

            // generate 100 seed points
            /* 
            generateRandomSeedPoints(10);

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
            */


            // SPLIT AND MERGE  
            // Apply region splitting and merging on the whole image
            namedWindow("Region Splitting and Merging Result", WINDOW_AUTOSIZE);
            // imshow("Region Splitting and Merging Result", rsm_result);
            // waitKey(0);
            Mat rsm1 = regionSplittingAndMerging(grayImage, Rect(0, 0, ct_slice.cols, ct_slice.rows), 5, rng);
            // Mat rsm2 = regionSplittingAndMerging(ct_slice, Rect(ct_slice.cols/2, 0, ct_slice.cols/2, ct_slice.rows/2), 5, rng);
            // Mat rsm3 = regionSplittingAndMerging(ct_slice, Rect(0, ct_slice.rows/2, ct_slice.cols/2, ct_slice.rows/2), 5, rng);
            // Mat rsm4 = regionSplittingAndMerging(ct_slice, Rect(ct_slice.cols/2, ct_slice.rows/2, ct_slice.cols/2, ct_slice.rows/2), 5, rng);
            // Mat rsm_result = rsm1 + rsm2 + rsm3 + rsm4;

            Mat coloredImage(ct_slice.size(), CV_8UC3);

            // Utiliser une table de hachage pour stocker les couleurs attribuées à chaque région
            unordered_map<int, Vec3b> colorMap;

            // Parcourir chaque pixel de l'image segmentée
            for (int i = 0; i < ct_slice.rows; ++i) {
                for (int j = 0; j < ct_slice.cols; ++j) {
                    // Récupérer la valeur de pixel de l'image segmentée (remplacer cela avec votre propre logique de segmentation)
                    int label = rsm1.at<uchar>(i, j);
                    for(int k = -20; k <= 20; k++ ) {
                        if (colorMap.find(label + k) != colorMap.end()){
                            label = label + k;
                            break;
                        } 
                    }

                    // Vérifier si la couleur a déjà été attribuée à cette région
                    if (colorMap.find(label) == colorMap.end()) {
                        // Si la couleur n'a pas encore été attribuée, attribuer une couleur aléatoire
                        colorMap[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
                    }

                    // Attribuer la couleur de la région au pixel correspondant dans l'image résultante
                    coloredImage.at<Vec3b>(i, j) = colorMap[label];
                }
            }

            imshow("Region Splitting and Merging Result", coloredImage );
            waitKey(0);

            Mat borders;
            Canny(coloredImage, borders, 30, 60);
            imshow("Region Borders", borders);
            waitKey(0);

            destroyAllWindows();
        }

        /**
         * Destructor for the RegionGrowing class.
         * 
         * This destructor is empty as there are no explicit resource allocations
         * or clean-up operations to be performed when an instance of RegionGrowing is destroyed.
         */
        ~RegionGrowing()
        {
        }

        /**
         * Callback function for mouse events, specifically handling left button down events.
         * Adds the clicked point to the seed set and visualizes it on the image.
         * 
         * @param event The type of mouse event (e.g., EVENT_LBUTTONDOWN).
         * @param x The x-coordinate of the mouse click.
         * @param y The y-coordinate of the mouse click.
         * @param flags Additional flags indicating the state of the mouse buttons and modifier keys.
         * @param userdata A pointer to the RegionGrowing instance associated with the callback.
         */
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

        /**
         * Applies region growing algorithm starting from a seed point in the input image.
         * 
         * @param anImage The input image to be segmented.
         * @param seedPoint The seed point for the region growing algorithm.
         * @param anInValue The value to assign to the segmented region in the output matrix.
         * @param tolerance The intensity difference tolerance for region growing.
         * @return The matrix representing the segmented region using region growing.
         */
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
