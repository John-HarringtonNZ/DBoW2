/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <experimental/filesystem>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features, const std::string &img_path, vector<string> &img_names);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &memory_imgs, const vector<vector<cv::Mat > > &target_imgs, const vector<string> &memory_img_names, const vector<string> &target_img_names);
void testDatabase(
  const vector<vector<cv::Mat > > &memory_imgs,
  const vector<vector<cv::Mat > > &target_imgs,
  const vector<string> &memory_img_names,
  const vector<string> &target_img_names,
  int top_n
);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cout << "Usage: ./demo <MEMORY_DIR> <TARGET_IMG_DIR> <TOP_N>\n";
    return -1;
  }
  std::string memory_dir = argv[1];
  std::string target_dir = argv[2];
  int top_n = stoi(argv[3]);

  vector<vector<cv::Mat > > memory;
  vector<string> memory_img_names;
  loadFeatures(memory, memory_dir, memory_img_names);
  vector<vector<cv::Mat> > targets;
  vector<string> target_img_names;
  loadFeatures(targets, target_dir, target_img_names);

  testVocCreation(memory, targets, memory_img_names, target_img_names);

  // wait();

  testDatabase(memory, targets, memory_img_names, target_img_names, top_n);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, const std::string &img_path, vector<string> &img_names)
{
  features.clear();
  img_names.clear();

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (const auto & entry : std::experimental::filesystem::directory_iterator(img_path)) {
    std::string img_name = entry.path();
    std::cout << "Found img: " << img_name << "\n";
    img_names.push_back(img_name);
    cv::Mat image = cv::imread(img_name, 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(
  const vector<vector<cv::Mat > > &memory_imgs,
  const vector<vector<cv::Mat > > &target_imgs,
  const vector<string> &memory_img_names,
  const vector<string> &target_img_names)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(memory_imgs);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  // BowVector v1, v2;
  // for(unsigned int i = 0; i < target_imgs.size(); i++)
  // {
  //   voc.transform(target_imgs[i], v1);
  //   for(unsigned int j = 0; j < memory_imgs.size(); j++)
  //   {
  //     voc.transform(memory_imgs[j], v2);
      
  //     double score = voc.score(v1, v2);
  //     cout << "Target " << target_img_names[i] << " vs Memory " << memory_img_names[j] << ": " << score << endl;
  //   }
  // }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(
  const vector<vector<cv::Mat > > &memory_imgs,
  const vector<vector<cv::Mat > > &target_imgs,
  const vector<string> &memory_img_names,
  const vector<string> &target_img_names,
  int top_n
)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(unsigned int i = 0; i < memory_imgs.size(); i++)
  {
    db.add(memory_imgs[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(unsigned int i = 0; i < target_imgs.size(); i++)
  {
    db.query(target_imgs[i], ret, top_n);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Target " << target_img_names[i] << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


