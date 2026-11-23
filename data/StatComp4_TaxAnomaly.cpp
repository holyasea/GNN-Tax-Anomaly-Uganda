/*
 * StatComp4_TaxAnomaly.cpp
 * STA 8103 Statistical Computing IV - Semester II 2026/27
 * Student: Asea Holyweek | 2500721983 | 2025/HD06/21983U
 *
 * PURPOSE:
 *   Menu-driven C++ program that models the Uganda EFRIS
 *   invoice network as a directed graph and detects anomalous
 *   taxpayers using an Isolation Forest approach.
 *
 * OOP CONCEPTS USED:
 *   - Encapsulation  : each class hides its own data
 *   - Inheritance    : IsolationTree and IsolationForest
 *                      inherit from BaseDetector
 *   - Polymorphism   : virtual score() function overridden
 *                      in each subclass
 *   - Abstraction    : user interacts through a menu; internal
 *                      tree logic is hidden
 *
 * DATA STRUCTURES USED:
 *   - Directed Graph      : adjacency list (unordered_map)
 *   - Hash Map            : TIN string -> integer node index
 *   - Binary Decision Tree: IsolationTree nodes (struct)
 *   - Vector<double>      : feature vectors per taxpayer
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <limits>
#include <memory>

using namespace std;

// ============================================================
// CONSTANTS (match actual research parameters)
// ============================================================
const int    NUM_FEATURES   = 10;   // feature_cols in R script
const int    NUM_TREES      = 200;  // trees at convergence
const double ANOMALY_CUTOFF = 0.10; // top 10% flagged

// ============================================================
// Taxpayer feature vector (10 features from EFRIS data)
// ============================================================
struct Taxpayer {
    string tin;          // Taxpayer Identification Number
    int    nodeIndex;    // integer index 0..N-1
    double features[NUM_FEATURES];
    /*
     * features[0]  = total_sales
     * features[1]  = total_purchases
     * features[2]  = output_input_vat_ratio
     * features[3]  = n_unique_partners
     * features[4]  = avg_invoice_value
     * features[5]  = var_sale_value
     * features[6]  = var_purchase_value
     * features[7]  = purchase_to_sales_ratio
     * features[8]  = n_months_active_sales
     * features[9]  = n_months_active_purch
     */
    double anomalyScore; // final Isolation Forest score
};

// ============================================================
// BASE CLASS: abstract detector interface (Abstraction)
// ============================================================
class BaseDetector {
public:
    // Pure virtual: every subclass MUST implement this
    virtual double score(const Taxpayer& tp) const = 0;
    virtual ~BaseDetector() {}
};

// ============================================================
// ISOLATION TREE NODE (internal binary tree node)
// ============================================================
struct TreeNode {
    bool   isLeaf;
    int    splitFeature; // which of the 10 features to split on
    double splitValue;   // random threshold
    int    size;         // number of records at this node
    unique_ptr<TreeNode> left;
    unique_ptr<TreeNode> right;

    // Leaf constructor
    TreeNode(int sz) :
        isLeaf(true), splitFeature(-1),
        splitValue(0.0), size(sz) {}

    // Internal node constructor
    TreeNode(int feat, double val) :
        isLeaf(false), splitFeature(feat),
        splitValue(val), size(0) {}
};

// ============================================================
// CLASS: IsolationTree -- ONE binary decision tree
//        Inherits from BaseDetector (Inheritance)
// ============================================================
class IsolationTree : public BaseDetector {
private:
    unique_ptr<TreeNode> root;
    int maxDepth;
    mt19937 rng;

    // Recursive tree builder
    unique_ptr<TreeNode> buildTree(
        const vector<Taxpayer*>& pts, int depth)
    {
        int n = (int)pts.size();
        // Stop: leaf reached if only 1 point or depth limit hit
        if (n <= 1 || depth >= maxDepth)
            return make_unique<TreeNode>(n);

        // Pick a random feature and a random split value
        uniform_int_distribution<int> featDist(0, NUM_FEATURES-1);
        int feat = featDist(rng);

        // Find min and max of this feature across current points
        double mn =  1e18, mx = -1e18;
        for (auto* p : pts) {
            mn = min(mn, p->features[feat]);
            mx = max(mx, p->features[feat]);
        }
        if (mn >= mx) return make_unique<TreeNode>(n); // all same

        uniform_real_distribution<double> valDist(mn, mx);
        double val = valDist(rng);

        // Partition records into left (<= val) and right (> val)
        vector<Taxpayer*> leftPts, rightPts;
        for (auto* p : pts) {
            if (p->features[feat] <= val) leftPts.push_back(p);
            else                          rightPts.push_back(p);
        }

        auto node = make_unique<TreeNode>(feat, val);
        node->left  = buildTree(leftPts,  depth+1);
        node->right = buildTree(rightPts, depth+1);
        return node;
    }

    // Walk one record down the tree; return path length
    double pathLength(const Taxpayer& tp,
                      const TreeNode* node, int depth) const
    {
        if (!node || node->isLeaf) {
            // Correction for unbuilt subtree depth
            int sz = node ? node->size : 1;
            return depth + averagePathLength(sz);
        }
        if (tp.features[node->splitFeature] <= node->splitValue)
            return pathLength(tp, node->left.get(),  depth+1);
        else
            return pathLength(tp, node->right.get(), depth+1);
    }

    // Expected path length for a dataset of size n
    static double averagePathLength(int n) {
        if (n <= 1) return 0.0;
        double h = log((double)n - 1.0) + 0.5772156649; // Euler
        return 2.0 * h - (2.0 * (n - 1.0) / n);
    }

public:
    // Constructor: build tree from sample of taxpayers
    IsolationTree(const vector<Taxpayer*>& pts,
                  int depth, unsigned seed)
        : maxDepth(depth), rng(seed)
    {
        root = buildTree(pts, 0);
    }

    // Polymorphism: override base class score()
    double score(const Taxpayer& tp) const override {
        return pathLength(tp, root.get(), 0);
    }
};

// ============================================================
// CLASS: IsolationForest -- ENSEMBLE of IsolationTree objects
//        Inherits from BaseDetector (Inheritance)
// ============================================================
class IsolationForest : public BaseDetector {
private:
    vector<unique_ptr<IsolationTree>> trees;
    int    numTrees;
    int    sampleSize;
    double avgPathLen; // c(n) normalizer

public:
    IsolationForest(int nTrees, int sample)
        : numTrees(nTrees), sampleSize(sample)
    {
        // c(n): expected path length for sample size
        double h = log((double)sample - 1.0) + 0.5772156649;
        avgPathLen = 2.0 * h - (2.0 * (sample - 1.0) / sample);
    }

    // Build all trees from a taxpayer dataset
    void fit(vector<Taxpayer>& taxpayers) {
        mt19937 masterRng(7142); // fixed seed from research
        uniform_int_distribution<int>
            idxDist(0, (int)taxpayers.size()-1);
        int maxDepth = (int)ceil(log2((double)sampleSize));

        for (int t = 0; t < numTrees; t++) {
            // Draw a subsample without replacement
            vector<Taxpayer*> sample;
            vector<int> chosen;
            while ((int)sample.size() < sampleSize &&
                   (int)sample.size() < (int)taxpayers.size())
            {
                int idx = idxDist(masterRng);
                // Simple reservoir: skip duplicates
                if (find(chosen.begin(),chosen.end(),idx)
                        == chosen.end()) {
                    chosen.push_back(idx);
                    sample.push_back(&taxpayers[idx]);
                }
            }
            unsigned seed = masterRng();
            trees.push_back(
                make_unique<IsolationTree>(sample, maxDepth, seed)
            );
        }
    }

    // Polymorphism: override base class score()
    // Returns anomaly score in [0,1]; closer to 1 = more unusual
    double score(const Taxpayer& tp) const override {
        double sumPath = 0.0;
        for (const auto& tree : trees)
            sumPath += tree->score(tp);
        double avgPath = sumPath / trees.size();
        return pow(2.0, -avgPath / avgPathLen);
    }
};

// ============================================================
// CLASS: TaxpayerGraph -- Directed commercial network
//        (Encapsulation: internal adjacency list is private)
// ============================================================
class TaxpayerGraph {
private:
    // Hash map: TIN string -> integer node index  O(1) lookup
    unordered_map<string, int>  tinToIndex;
    // Adjacency list: nodeIndex -> list of neighbor indices
    vector<vector<int>>         adjList;
    // All taxpayer records
    vector<Taxpayer>            nodes;
    int                         nodeCount;

public:
    TaxpayerGraph() : nodeCount(0) {}

    // Add a taxpayer node (encapsulated)
    int addNode(const string& tin,
                const double feats[NUM_FEATURES])
    {
        if (tinToIndex.count(tin)) return tinToIndex[tin];
        int idx = nodeCount++;
        tinToIndex[tin] = idx;
        adjList.push_back({});

        Taxpayer tp;
        tp.tin = tin;
        tp.nodeIndex = idx;
        tp.anomalyScore = 0.0;
        for (int f = 0; f < NUM_FEATURES; f++)
            tp.features[f] = feats[f];
        nodes.push_back(tp);
        return idx;
    }

    // Add a directed trade edge (seller -> buyer)
    void addEdge(int sellerIdx, int buyerIdx) {
        if (sellerIdx < nodeCount && buyerIdx < nodeCount)
            adjList[sellerIdx].push_back(buyerIdx);
    }

    // Return number of nodes and edges
    int getNodeCount() const { return nodeCount; }
    int getEdgeCount() const {
        int total = 0;
        for (const auto& nb : adjList) total += (int)nb.size();
        return total;
    }

    // Network density
    double getDensity() const {
        if (nodeCount < 2) return 0.0;
        long long maxE = (long long)nodeCount * (nodeCount - 1);
        return (double)getEdgeCount() / maxE;
    }

    // Return modifiable taxpayer list (for forest fitting)
    vector<Taxpayer>& getTaxpayers() { return nodes; }

    // Print network summary
    void printSummary() const {
        cout << "\n--- Commercial Network Summary ---\n";
        cout << "  Businesses (nodes) : " << nodeCount << "\n";
        cout << "  Trade links (edges): " << getEdgeCount() << "\n";
        cout << fixed << setprecision(6);
        cout << "  Network density    : " << getDensity() << "\n";
    }

    // Print top K anomalous taxpayers
    void printTopAnomalies(int topK) const {
        // Copy and sort by anomaly score descending
        vector<Taxpayer> sorted = nodes;
        sort(sorted.begin(), sorted.end(),
             [](const Taxpayer& a, const Taxpayer& b){
                 return a.anomalyScore > b.anomalyScore;
             });
        cout << "\n--- Top " << topK
             << " Anomalous Taxpayers ---\n";
        cout << left << setw(12) << "Node"
             << setw(20) << "TIN"
             << setw(14) << "Anomaly Score"
             << setw(14) << "Total Sales"
             << "VAT Ratio\n";
        cout << string(74, '-') << "\n";
        for (int i = 0; i < min(topK,(int)sorted.size()); i++){
            cout << left << setw(12) << sorted[i].nodeIndex
                 << setw(20) << sorted[i].tin
                 << fixed << setprecision(4)
                 << setw(14) << sorted[i].anomalyScore
                 << setw(14) << sorted[i].features[0]
                 << sorted[i].features[2] << "\n";
        }
    }
};

// ============================================================
// HELPER: Load demonstration dataset (mimics EFRIS structure)
//   In the real research, this was loaded from CSV files using
//   R's data.table. Here we hard-code a small representative
//   sample for program demonstration purposes.
// ============================================================
void loadDemoData(TaxpayerGraph& graph) {
    /*
     * 10 features per taxpayer (in order of feature_cols):
     * [0] total_sales  [1] total_purchases  [2] vat_ratio
     * [3] unique_partners  [4] avg_invoice  [5] var_sales
     * [6] var_purchases  [7] purch_to_sales
     * [8] months_active_sales  [9] months_active_purch
     *
     * Values are scaled to billions of UGX for readability.
     * Taxpayers 8 and 9 are seeded to look anomalous:
     *   - Zero sales, large purchases, extreme VAT ratio.
     */
    struct RawRecord {
        const char* tin;
        double f[NUM_FEATURES];
    };

    RawRecord records[] = {
        // NORMAL TAXPAYERS
        {"TIN-001",{45.2,38.1,0.85,12,3.77,2.1,1.8,0.84,48,48}},
        {"TIN-002",{82.3,70.5,0.92,25,3.29,5.4,4.2,0.86,48,48}},
        {"TIN-003",{30.0,27.2,0.88,8, 3.75,1.5,1.2,0.91,36,40}},
        {"TIN-004",{120.5,99.8,0.94,40,3.01,8.7,7.1,0.83,48,48}},
        {"TIN-005",{15.7,13.4,0.87,6, 2.62,0.8,0.7,0.85,30,32}},
        {"TIN-006",{67.4,55.2,0.91,18,3.74,4.0,3.5,0.82,48,48}},
        {"TIN-007",{54.8,46.1,0.90,15,3.65,3.2,2.8,0.84,44,44}},
        // ANOMALOUS TAXPAYERS (mimic consensus cases in study)
        // Node 8: zero sales, claims 119B in purchases (like 17
        //         zero-sales businesses found in real results)
        {"TIN-008",{0.0, 119.3,0.00,2, 59.65,0.0,980.2,999.9,
                    0,  24}},
        // Node 9: extreme VAT ratio, traded only 12 months
        {"TIN-009",{1.2, 88.5, 0.01,1, 88.5, 0.1,750.4,73.75,
                    12, 12}},
    };

    int n = (int)(sizeof(records)/sizeof(records[0]));
    for (int i = 0; i < n; i++)
        graph.addNode(records[i].tin, records[i].f);

    // Add some directed trade edges (seller -> buyer)
    // Edges chosen to mimic real EFRIS trading chains
    graph.addEdge(0,1); graph.addEdge(1,3);
    graph.addEdge(2,0); graph.addEdge(3,5);
    graph.addEdge(4,6); graph.addEdge(5,6);
    graph.addEdge(6,1);
    // Anomalous nodes connect suspiciously
    graph.addEdge(7,0); graph.addEdge(8,3);
}

// ============================================================
// MENU FUNCTIONS
// ============================================================
void runAnomalyDetection(TaxpayerGraph& graph,
                         IsolationForest& forest)
{
    auto& taxpayers = graph.getTaxpayers();
    for (auto& tp : taxpayers)
        tp.anomalyScore = forest.score(tp);

    int topK = max(1, (int)ceil(ANOMALY_CUTOFF * taxpayers.size()));
    graph.printTopAnomalies(topK);

    // Count flagged (top 10%)
    vector<double> scores;
    for (auto& tp : taxpayers) scores.push_back(tp.anomalyScore);
    sort(scores.rbegin(), scores.rend());
    double threshold = scores[topK - 1];
    int flagged = 0;
    for (auto& tp : taxpayers)
        if (tp.anomalyScore >= threshold) flagged++;

    cout << "\n  Threshold score (top 10%)  : "
         << fixed << setprecision(4) << threshold << "\n";
    cout << "  Taxpayers flagged          : " << flagged << "\n";
    cout << "  (In full study: 62 of 614 flagged per model)\n";
}

void showResearchContext() {
    cout << "\n=== Research Context (Full EFRIS Study) ===\n";
    cout << " Data period             : 2018 to 2022\n";
    cout << " Total invoices          : 6,097,667\n";
    cout << " Sales invoices          : 1,048,575\n";
    cout << " Purchase invoices       : 5,049,092\n";
    cout << " Business nodes          : 30,173\n";
    cout << " Trade links (edges)     : 144,426\n";
    cout << " Network density         : 0.000159\n";
    cout << " Louvain communities     : 16 (Q = 0.3474)\n";
    cout << " Focal large taxpayers   : 614\n";
    cout << " Isolation Forest trees  : 200\n";
    cout << " R-GCN training time     : ~30.9 seconds\n";
    cout << " GAT training time       : ~70.8 seconds\n";
    cout << " Flagged per model       : 62 taxpayers (top 10%)\n";
    cout << " Consensus anomalies     : 28 (flagged by 2+ models)\n";
    cout << " With zero sales         : 17 (claimed 119.3B UGX)\n";
    cout << " Revenue at risk (5%)    : 6.0B UGX\n";
    cout << " Revenue at risk (25%)   : 29.8B UGX\n";
    cout << " Revenue at risk (50%)   : 59.6B UGX\n";
}

void showAlgorithmInfo() {
    cout << "\n=== Algorithms Used ===\n";
    cout << "1. Louvain Community Detection\n";
    cout << "   Maximizes modularity Q to find 16 trade clusters.\n";
    cout << "   Result: Q = 0.3474 on 30,173 nodes.\n\n";
    cout << "2. Message Passing (GNN)\n";
    cout << "   Each business collects financial summaries from\n";
    cout << "   its trade neighbors across 2 network layers.\n\n";
    cout << "3. Negative Edge Sampling\n";
    cout << "   20,000 non-trading pairs sampled per epoch to\n";
    cout << "   teach the model what a real trade link looks like.\n\n";
    cout << "4. K-Means Clustering (k=7, silhouette=0.697)\n";
    cout << "   Groups normal business GAT embeddings into 7\n";
    cout << "   trading styles; outliers are anomaly candidates.\n\n";
    cout << "5. Isolation Forest (200 trees)\n";
    cout << "   Binary trees that isolate unusual records early.\n";
    cout << "   Score near 1.0 = highly anomalous.\n";
}

void showOOPConcepts() {
    cout << "\n=== OOP Concepts Demonstrated ===\n";
    cout << "1. Encapsulation\n";
    cout << "   TaxpayerGraph hides its adjacency list and hash\n";
    cout << "   map. Users interact only through public methods.\n\n";
    cout << "2. Inheritance\n";
    cout << "   IsolationTree  --> inherits BaseDetector\n";
    cout << "   IsolationForest --> inherits BaseDetector\n";
    cout << "   Both gain the score() interface automatically.\n\n";
    cout << "3. Polymorphism\n";
    cout << "   BaseDetector* pointer can hold either tree type.\n";
    cout << "   The correct score() runs without if/else checks.\n\n";
    cout << "4. Abstraction\n";
    cout << "   Menu hides all tree-building internals from user.\n";
    cout << "   User just selects 'Run Detection' and sees output.\n";
}

// ============================================================
// MAIN -- Menu-driven entry point
// ============================================================
int main() {
    cout << "========================================\n";
    cout << " STA 8103 Statistical Computing IV\n";
    cout << " Tax Anomaly Detection -- EFRIS Network\n";
    cout << " Asea Holyweek | 2500721983\n";
    cout << "========================================\n";

    // Build the commercial graph
    TaxpayerGraph graph;
    loadDemoData(graph);

    // Build and fit the Isolation Forest
    int sampleSz = min(256, graph.getNodeCount());
    IsolationForest forest(NUM_TREES, sampleSz);
    forest.fit(graph.getTaxpayers());

    int choice = 0;
    do {
        cout << "\n============= MAIN MENU =============\n";
        cout << "1. Show network summary\n";
        cout << "2. Run anomaly detection (Isolation Forest)\n";
        cout << "3. Show full research context (EFRIS study)\n";
        cout << "4. Show algorithms used\n";
        cout << "5. Show OOP concepts demonstrated\n";
        cout << "0. Exit\n";
        cout << "Enter choice: ";
        cin  >> choice;

        switch (choice) {
            case 1: graph.printSummary();                   break;
            case 2: runAnomalyDetection(graph, forest);     break;
            case 3: showResearchContext();                   break;
            case 4: showAlgorithmInfo();                    break;
            case 5: showOOPConcepts();                      break;
            case 0: cout << "Exiting. Goodbye!\n";         break;
            default: cout << "Invalid choice. Try again.\n"; break;
        }
    } while (choice != 0);

    return 0;
}

