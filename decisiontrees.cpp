#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <cmath>
#include <pthread.h>
#include <atomic>


using namespace std;

struct DecisionTreeNode {
    int featureIndex;
    float threshold;
    int label;
    DecisionTreeNode* left;
    DecisionTreeNode* right;
};

struct Point {
    vector<float> features;
    int label;
    Point(vector<float> features, int label) : features{features}, label{label} {}
};

struct ThreadArgs {
    int* threadIndex;
    int* numThreads;
    const vector<Point>* data;
    vector<float>* bestGini;
    vector<int>* bestFeatureIndex;
    vector<float>* bestThreshold;
};

float calculateGini(const vector<Point>& data) {
    int classCounts[2] = {0};
    for (const Point& point : data) {
        classCounts[point.label]++;
    }
    float gini = 1.0;
    for (int i = 0; i < 2; i++) {
        float p = classCounts[i] / (float)data.size();
        gini -= p * p;
    }
    return gini;
}

pair<vector<Point>, vector<Point> > splitData(const vector<Point>& data, int featureIndex, float threshold) {
    vector<Point> leftData;
    vector<Point> rightData;
    for (int i = 0; i < data.size(); i++) {
        if (data[i].features[featureIndex] < threshold + 0.0001) {
            leftData.push_back(data[i]);
        } else {
            rightData.push_back(data[i]);
        }
    }
    return make_pair(leftData, rightData);
}

pair <int, float> findBestSplit(const vector<Point>& data) {
    float bestGini = 1.0;
    int bestFeatureIndex = -1;
    float bestThreshold = 0.0;
    for (int featureIndex = 0; featureIndex < data[0].features.size(); featureIndex++) {
        for (int i = 0; i < data.size(); i++) {
            float threshold = data[i].features[featureIndex];
            pair<vector<Point>, vector<Point> > split = splitData(data, featureIndex, threshold);
            float gini = calculateGini(split.first) * (split.first.size() / (data.size() + 0.0)) + calculateGini(split.second) * (split.second.size() / (data.size() + 0.0));
            if (gini < bestGini) {
                bestGini = gini;
                bestFeatureIndex = featureIndex;
                bestThreshold = threshold;
            }
        }
    }
    return make_pair(bestFeatureIndex, bestThreshold);
}

void* findBestSplitParallelThread(void* args) {
    ThreadArgs* threadArgs = static_cast<ThreadArgs*>(args);
    int threadIndex = *(threadArgs->threadIndex);
    int numThreads = *(threadArgs->numThreads);
    for (int featureIndex = threadIndex; featureIndex < (*(threadArgs->data))[0].features.size(); featureIndex+= numThreads) {
        for (int i = 0; i < threadArgs->data->size(); i++) {
            float threshold = (*(threadArgs->data))[i].features[featureIndex];
            pair<vector<Point>, vector<Point> > split = splitData((*(threadArgs->data)), featureIndex, threshold);
            float gini = calculateGini(split.first) * (split.first.size() / (threadArgs->data->size() + 0.0)) + calculateGini(split.second) * (split.second.size() / (threadArgs->data->size() + 0.0));
            if (gini < (*(threadArgs->bestGini))[threadIndex]) {
                (*(threadArgs->bestGini))[threadIndex] = gini;
                (*(threadArgs->bestFeatureIndex))[threadIndex] = featureIndex;
                (*(threadArgs->bestThreshold))[threadIndex] = threshold;
            }
        }
    }
    return NULL;
}

pair <int, float> findBestSplitParallel(const vector<Point>& data, const int numThreads) {
    float bestGini = 1.0;
    int bestFeatureIndex = -1;
    float bestThreshold = 0.0;
    vector<float> bestGinis(numThreads, 1.0);
    vector<int> bestFeatureIndices(numThreads, -1);
    vector<float> bestThresholds(numThreads, 0.0);
    pthread_t threads[numThreads];
    for (int threadIndex = 0; threadIndex < numThreads; threadIndex++) {
        pthread_create(&threads[threadIndex], NULL, findBestSplitParallelThread, new ThreadArgs{new int(threadIndex), new int(numThreads), &data, &bestGinis, &bestFeatureIndices, &bestThresholds});
    }
    for (int threadIndex = 0; threadIndex < numThreads; threadIndex++) {
        pthread_join(threads[threadIndex], NULL);
    }
    for (int threadIndex = 0; threadIndex < numThreads; threadIndex++) {
        if (bestGinis[threadIndex] < bestGini) {
            bestGini = bestGinis[threadIndex];
            bestFeatureIndex = bestFeatureIndices[threadIndex];
            bestThreshold = bestThresholds[threadIndex];
        }
    }
    return make_pair(bestFeatureIndex, bestThreshold);
}

DecisionTreeNode* sequentialBuildDecisionTree(const vector<Point>& data, const int depthLeft) {
    if (data.size() == 0) {
        return NULL;
    }
    int classCounts[2] = {0};
    for (const Point& point : data) {
        classCounts[point.label]++;
    }
    if (classCounts[0] == data.size()) {
        return new DecisionTreeNode{-1, 0.0, 0, NULL, NULL};
    }
    if (classCounts[1] == data.size()) {
        return new DecisionTreeNode{-1, 0.0, 1, NULL, NULL};
    }
    //pair<int, float> bestSplit = findBestSplit(data);
    pair<int, float> bestSplit = findBestSplitParallel(data, 2);
    if (depthLeft == 0 || bestSplit.first == -1) {
        int majority = classCounts[0] > classCounts[1] ? 0 : 1;
        return new DecisionTreeNode{-1, 0.0, majority, NULL, NULL};
    }
    pair<vector<Point>, vector<Point> > split = splitData(data, bestSplit.first, bestSplit.second);
    DecisionTreeNode* left = sequentialBuildDecisionTree(split.first, depthLeft - 1);
    DecisionTreeNode* right = sequentialBuildDecisionTree(split.second, depthLeft - 1);
    return new DecisionTreeNode{bestSplit.first, bestSplit.second, -1, left, right};
}

int predict(const DecisionTreeNode* root, const Point& point) {
    if (root->label != -1) {
        return root->label;
    }
    if (point.features[root->featureIndex] < root->threshold + 0.0001) {
        return predict(root->left, point);
    } else {
        return predict(root->right, point);
    }
}

// Hard coded to feature/label files for prostate cancer genome dataset
void load_dataset(vector<Point>& data){
    fstream fin;
    queue<int> label_queue;
    fin.open("prostate2.txt", ios::in);
    int cur_label;
    while(fin >> cur_label) {
        label_queue.push(cur_label);
    }
    fin.close(); 
    fin.open("prostate1.csv", ios::in);
    string feature_name_line;
    fin >> feature_name_line;
    string feature_line;
    while(fin >> feature_line) {
        vector<float>feature_vec;
        stringstream feature_ss(feature_line);
        float iter;
        while(feature_ss >> iter) {
            feature_vec.push_back(iter);
            if(feature_ss.peek() == ',') {
                feature_ss.ignore();
            }
        }
        data.push_back(Point(feature_vec, label_queue.front()));
        label_queue.pop();
    }
}

int main() {
    vector<Point> data;
    load_dataset(data);
    DecisionTreeNode* root = sequentialBuildDecisionTree(data, 2);
    //cout << (predict(root, data[0]) == 0) << endl;
    return 0;
}