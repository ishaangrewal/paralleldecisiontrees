#include <iostream>
#include <vector>
#include <cmath>

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
};

float calculateGini(const vector<Point>&data) {
    if (data.size() == 0) {
        return 0.0;
    }
    int classCounts[2] = {0};
    for (const Point& point : data) {
        classCounts[point.label]++;
    }

    float gini = 1.0;
    for (int i = 0; i < 2; ++i) {
        float probability = static_cast<float>(classCounts[i]) / data.size();
        gini -= probability * probability;
    }

    return gini;
}

pair<vector<Point>, vector<Point> > splitData(const vector<Point>& data, int featureIndex, float threshold) {
    vector<Point> leftData;
    vector<Point> rightData;
    for (int i = 0; i < data.size(); i++) {
        if (data[i].features[featureIndex] < threshold) {
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
            float gini = calculateGini(split.first) * (split.first.size() / data.size()) + calculateGini(split.second) * (split.second.size() / data.size());
            if (gini < bestGini) {
                bestGini = gini;
                bestFeatureIndex = featureIndex;
                bestThreshold = threshold;
            }
        }
    }
    return make_pair(bestFeatureIndex, bestThreshold);
}




DecisionTreeNode* sequentialBuildDecisionTree(const vector<Point>& data) {
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
    pair<int, float> bestSplit = findBestSplit(data);
    if (bestSplit.first == -1) {
        int majority = classCounts[0] > classCounts[1] ? 0 : 1;
        return new DecisionTreeNode{-1, 0.0, majority, NULL, NULL};
    }
    pair<vector<Point>, vector<Point> > split = splitData(data, bestSplit.first, bestSplit.second);
    DecisionTreeNode* left = sequentialBuildDecisionTree(split.first);
    DecisionTreeNode* right = sequentialBuildDecisionTree(split.second);
    return new DecisionTreeNode{bestSplit.first, bestSplit.second, -1, left, right};
}
int main() {
    return 0;
}