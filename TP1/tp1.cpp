#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace std;
using namespace cv;

class SplitAndMerge
{
    private:
        Mat ct_slice;
        Mat colour_ct_slice;
        RNG rng;

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


    public:

        /**
         * Constructor for the SplitAndMerge class.
         * 
         * Initializes the instance with an input color image, generates seed points,
         * performs region growing segmentation, and displays the results.
         * 
         * @param image_path Path to the input color image.
         */
        SplitAndMerge(String image_path)
        {
            
            ct_slice = imread(image_path, IMREAD_COLOR);
            rng = RNG(12345);

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


            // SPLIT AND MERGE  
            // Apply region splitting and merging on the whole image
            namedWindow("Region Splitting and Merging Result", WINDOW_AUTOSIZE);
            // imshow("Region Splitting and Merging Result", rsm_result);
            // waitKey(0);
            Mat rsm1 = regionSplittingAndMerging(grayImage, Rect(0, 0, ct_slice.cols, ct_slice.rows), 2, rng);
            // Mat rsm2 = regionSplittingAndMerging(ct_slice, Rect(ct_slice.cols/2, 0, ct_slice.cols/2, ct_slice.rows/2), 5, rng);
            // Mat rsm3 = regionSplittingAndMerging(ct_slice, Rect(0, ct_slice.rows/2, ct_slice.cols/2, ct_slice.rows/2), 5, rng);
            // Mat rsm4 = regionSplittingAndMerging(ct_slice, Rect(ct_slice.cols/2, ct_slice.rows/2, ct_slice.cols/2, ct_slice.rows/2), 5, rng);
            // Mat rsm_result = rsm1 + rsm2 + rsm3 + rsm4;

            imshow("Region Splitting and Merging Before RandomColor", rsm1);


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
         * Destructor for the SplitAndMerge class.
         * 
         * This destructor is empty as there are no explicit resource allocations
         * or clean-up operations to be performed when an instance of SplitAndMerge is destroyed.
         */
        ~SplitAndMerge()
        {
        }

        
};




int main(int argc, char **argv)
{
    SplitAndMerge sm1("data/000009.jpg");
    SplitAndMerge sm2("data/000451.jpg");

    return 0;
}
