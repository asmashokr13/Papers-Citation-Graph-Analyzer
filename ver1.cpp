#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iomanip>

using namespace std;

//  SINGLY-LINKED LIST  (replaces vector<int> in loadEdges)

struct IntNode {
    int val;
    IntNode* next;
    IntNode(int v, IntNode* n = nullptr) : val(v), next(n) {}
};

class IntList {
public:
    IntNode* head = nullptr;
    IntNode* tail = nullptr;
    int size       = 0;

    void push_back(int v) {
        IntNode* node = new IntNode(v);
        if (!tail) head = tail = node;
        else       { tail->next = node; tail = node; }
        ++size;
    }

    ~IntList() {
        IntNode* cur = head;
        while (cur) {
          IntNode* nxt = cur->next; delete cur; cur = nxt; }
    }
};

//  FIXED-SIZE RESULT ARRAYS

constexpr int MAX_K     = 10000;   // max results for top-K
constexpr int MAX_SIM   = 10000;   // max candidates for similarity

struct TopResult {
    int rank;
    int id;
    int citations;
    string title;
};

struct SimilarResult {
    int    id      = 0;
    int    common  = 0;
    double jaccard = 0.0;
    string title;
};

//  GRAPH CLASS

class CitationGraph {
public:
    unordered_map<int, unordered_set<int>> adj;
    unordered_map<int, unordered_set<int>> radj;

    void addEdge(int from, int to) {
        adj[from].emplace(to);
        radj[to].emplace(from);
    }

    int inDegree(int node) const {
        auto it = radj.find(node);
        return (it != radj.end()) ? static_cast<int>(it->second.size()) : 0;
    }

    int outDegree(int node) const {
        auto it = adj.find(node);
        return (it != adj.end()) ? static_cast<int>(it->second.size()) : 0;
    }

    int edgeCount() const {
        int total = 0;
        for (auto& [u, vs] : adj) total += static_cast<int>(vs.size());
        return total;
    }

    // BFS
    unordered_set<int> bfsReachable(int start, int maxHops = 2) const {
        unordered_set<int> visited;
        queue<pair<int,int>> q;
        q.emplace(start, 0);
        visited.emplace(start);

        while (!q.empty()) {
            auto [node, depth] = q.front(); q.pop();
            if (depth >= maxHops) continue;
            auto it = adj.find(node);
            if (it == adj.end()) continue;
            for (int nbr : it->second) {
                if (!visited.count(nbr)) {
                    visited.emplace(nbr);
                    q.emplace(nbr, depth + 1);
                }
            }
        }
        visited.erase(start);
        return visited;
    }

    // common citations between 2 papers
    // time: O(min(|A|,|B|))
    int commonCitations(int a, int b) const {
        auto itA = adj.find(a);
        auto itB = adj.find(b);
        if (itA == adj.end() || itB == adj.end()) return 0;

        const unordered_set<int>* small = &itA->second;
        const unordered_set<int>* large = &itB->second;
        if (small->size() > large->size()) swap(small, large);

        int count = 0;
        for (int x : *small)
            if (large->count(x)) ++count;
        return count;
    }
};

//  DATA LOADING

using PaperMap = unordered_map<int, string>;

bool parseNodeLine(const string& line, int& id, string& title) {
    size_t i = 0;
    while (i < line.size() && isspace(static_cast<unsigned char>(line[i]))) ++i;  //3shan ignore el spaces
    if (i >= line.size() || !isdigit(static_cast<unsigned char>(line[i]))) return false;

    id = 0;
    while (i < line.size() && isdigit(static_cast<unsigned char>(line[i])))
        id = id * 10 + (line[i++] - '0'); //3shan convert mn el string ll int

    if (i >= line.size() || (line[i] != ' ' && line[i] != ',')) return false; //lazm b3d el num , or space
    ++i;

    string raw = line.substr(i);
    size_t cp = raw.find(",,");
    if (cp != string::npos) raw = raw.substr(0, cp);

    title.clear();
    for (char c : raw) title += (c == '_' ? ' ' : c);
    while (!title.empty() && (title.back() == ' ' || title.back() == ','))
        title.pop_back();
    return !title.empty();
}

// Load paper titles — Time: O(N)
PaperMap loadNodes(const string& path) {
    PaperMap papers;
    ifstream f(path);
    if (!f) { cerr << "Cannot open: " << path << "\n"; return papers; }
    string line;
    int skip = 0;
    while (getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (skip < 2 && (line == "id" || line == "name")) { ++skip; continue; }
        int id; string title;
        if (parseNodeLine(line, id, title) && !papers.count(id))
            papers[id] = title;
    }
    return papers;
}

// Load edges into graph using a linked list for interim storage
// Time: O(E)
void loadEdges(const string& path, CitationGraph& graph) {
    ifstream f(path);
    if (!f) { cerr << "Cannot open: " << path << "\n"; return; }

    string line;
    IntList nums;           //linked list
    int skip = 0;

    while (getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (skip < 2 && (line == "paper_id" || line == "citation_id")) { ++skip; continue; }
        try {
            size_t idx;
            int val = stoi(line, &idx);
            if (idx == line.size() ||
                line.substr(idx).find_first_not_of(" \t\r") == string::npos)
                nums.push_back(val);
        } catch (...) {}
    }

    // alk the ll in pairs
    IntNode* cur = nums.head;
    while (cur && cur->next) {
        graph.addEdge(cur->val, cur->next->val);
        cur = cur->next->next;
    }
}
//  TOP K MOST CITED  (Max-Heap on in-degree)

using HeapEntry = pair<int,int>;
using MaxHeap   = priority_queue<HeapEntry>;

// Build heap — Time: O(N log N)
MaxHeap buildMaxHeap(const CitationGraph& graph) {
    MaxHeap heap;
    for (auto& [node, citedBy] : graph.radj)
        heap.emplace(static_cast<int>(citedBy.size()), node);
    return heap;
}

// Fill caller-supplied array, return actual count
// Time: O(K log N)
int getTopK(const MaxHeap& heapRef, int k, const PaperMap& papers,
            TopResult* out, int outSize)
{
    if (k > outSize) k = outSize;
    MaxHeap heap = heapRef;
    int rank = 1, cnt = 0;
    while (!heap.empty() && rank <= k) {
        auto [citations, pid] = heap.top(); heap.pop();
        string t = papers.count(pid) ? papers.at(pid)
                                     : "[Unknown " + to_string(pid) + "]";
        out[cnt++] = { rank++, pid, citations, t };
    }
    return cnt;
}

// ─────────────────────────────────────────────────────────────
//  CLOSEST PAPERS  (common citations + Jaccard)
//  Uses a fixed-size array + insertion-sort-style bounded insert
// ─────────────────────────────────────────────────────────────

constexpr int DEFAULT_THRESHOLD = 3;

// Comparison: higher common first, then higher Jaccard
bool simGreater(const SimilarResult& a, const SimilarResult& b) {
    if (a.common != b.common) return a.common > b.common;
    return a.jaccard > b.jaccard;
}

// Fill caller-supplied array sorted descending by (common, jaccard)
// Returns actual count written (≤ topN ≤ outSize)
// Time: O(N * |adj[query]|) average
int findClosest(
    int queryId,
    const CitationGraph& graph,
    const PaperMap& papers,
    SimilarResult* out,
    int outSize,
    int threshold = DEFAULT_THRESHOLD,
    int topN      = 10)
{
    if (topN > outSize) topN = outSize;
    if (!graph.adj.count(queryId) || graph.adj.at(queryId).empty()) return 0;

    const unordered_set<int>& qCit = graph.adj.at(queryId);
    int filled = 0;

    for (auto& [pid, ignored] : graph.adj) {
        if (pid == queryId) continue;

        int common = graph.commonCitations(queryId, pid);
        if (common < threshold) continue;

        int uni = static_cast<int>(qCit.size()) + graph.outDegree(pid) - common;
        double jac = (uni > 0) ? (static_cast<double>(common) / uni) : 0.0;
        string t = papers.count(pid) ? papers.at(pid)
                                     : "[Unknown " + to_string(pid) + "]";
        SimilarResult sr { pid, common, jac, t };

        if (filled < topN) {
            // Array not yet full: place then bubble up to sorted position
            out[filled++] = sr;
            for (int i = filled - 1; i > 0 && simGreater(out[i], out[i-1]); --i)
                swap(out[i], out[i-1]);
        } else if (simGreater(sr, out[filled - 1])) {
            // Better than the worst kept result: replace tail, re-sort tail upward
            out[filled - 1] = sr;
            for (int i = filled - 1; i > 0 && simGreater(out[i], out[i-1]); --i)
                swap(out[i], out[i-1]);
        }
    }
    return filled;
}

// ─────────────────────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────────────────────

int lookupPaper(const string& q, const PaperMap& papers) {
    bool allDigits = !q.empty();
    for (char c : q)
        if (!isdigit(static_cast<unsigned char>(c))) { allDigits = false; break; }
    if (allDigits) {
        int pid = stoi(q);
        return papers.count(pid) ? pid : -1;
    }
    string ql = q;
    transform(ql.begin(), ql.end(), ql.begin(),
              [](unsigned char c){ return static_cast<char>(tolower(c)); });
    for (auto& [pid, t] : papers) {
        string tl = t;
        transform(tl.begin(), tl.end(), tl.begin(),
                  [](unsigned char c){ return static_cast<char>(tolower(c)); });
        if (tl.find(ql) != string::npos) return pid;
    }
    return -1;
}

string trunc(const string& s, size_t n = 60) {
    return (s.size() <= n) ? s : s.substr(0, n - 1) + "...";
}

void printDiv(char c = '-', int n = 80) { cout << string(n, c) << "\n"; }

// ─────────────────────────────────────────────────────────────
//  INTERACTIVE CLI
// ─────────────────────────────────────────────────────────────

void printMenu() {
    cout << "\n";
    printDiv('=');
    cout << "       Papers Citation Graph - Menu\n";
    printDiv('=');
    cout << "  1. Top-K most cited papers\n";
    cout << "  2. Find closest papers (bonus - common citations)\n";
    cout << "  3. BFS: papers reachable from a paper\n";
    cout << "  4. Look up a paper by ID or title\n";
    cout << "  5. Dataset / graph statistics\n";
    cout << "  0. Exit\n";
    printDiv('=');
    cout << "Enter choice: ";
}

void runCLI(const PaperMap& papers,
            const CitationGraph& graph,
            const MaxHeap& heap) {
    cout << "\n";
    printDiv('=');
    cout << "  Papers Citation Graph\n";
    cout << "  Papers loaded: " << papers.size()
         << "  |  Edges: " << graph.edgeCount() << "\n";
    printDiv('=');

    // Stack-allocated result buffers (no heap vectors)
    static TopResult    topBuf[MAX_K];
    static SimilarResult simBuf[MAX_SIM];

    while (true) {
        printMenu();
        string ch; getline(cin, ch);
        if (ch.empty()) continue;

        // 0: Exit
        if (ch == "0") {
            cout << "Goodbye!\n";
            break;
        }

        // 1: Top-K
        else if (ch == "1") {
            cout << "How many top papers? [default 10]: ";
            string ks; getline(cin, ks);
            int k = 10;
            try { if (!ks.empty()) k = stoi(ks); } catch (...) {}
            if (k > MAX_K) { cout << "Capped at " << MAX_K << ".\n"; k = MAX_K; }

            int cnt = getTopK(heap, k, papers, topBuf, MAX_K);
            cout << "\nTop " << cnt << " most cited papers:\n";
            printDiv();
            cout << left << setw(6)  << "Rank"
                         << setw(11) << "Citations"
                         << setw(8)  << "ID"
                         << "Title\n";
            printDiv();
            for (int i = 0; i < cnt; ++i)
                cout << left << setw(6)  << topBuf[i].rank
                             << setw(11) << topBuf[i].citations
                             << setw(8)  << topBuf[i].id
                             << trunc(topBuf[i].title) << "\n";
            printDiv();
        }

        // 2: Closest papers
        else if (ch == "2") {
            cout << "Enter paper ID or title keyword: ";
            string q; getline(cin, q);
            int pid = lookupPaper(q, papers);
            if (pid == -1) { cout << "Paper not found.\n"; continue; }

            cout << "Min shared citations [default " << DEFAULT_THRESHOLD << "]: ";
            string ts; getline(cin, ts);
            int thr = DEFAULT_THRESHOLD;
            try { if (!ts.empty()) thr = stoi(ts); } catch (...) {}

            cout << "Max results [default 10]: ";
            string tops; getline(cin, tops);
            int topN = 10;
            try { if (!tops.empty()) topN = stoi(tops); } catch (...) {}
            if (topN > MAX_SIM) { cout << "Capped at " << MAX_SIM << ".\n"; topN = MAX_SIM; }

            cout << "\nQuery: [" << pid << "] "
                 << trunc(papers.count(pid) ? papers.at(pid) : "?", 70) << "\n";
            cout << "Out-degree (cites): " << graph.outDegree(pid)
                 << "  |  In-degree (cited by): " << graph.inDegree(pid) << "\n";
            cout << "Threshold >= " << thr << " shared citations\n\n";
            cout << "Computing...\n";

            int cnt = findClosest(pid, graph, papers, simBuf, MAX_SIM, thr, topN);
            if (cnt == 0) {
                cout << "No papers share >= " << thr
                     << " citations. Try lowering the threshold.\n";
            } else {
                cout << "Found " << cnt << " close paper(s):\n";
                printDiv();
                cout << left << setw(9)  << "Common"
                             << setw(10) << "Jaccard"
                             << setw(8)  << "ID"
                             << "Title\n";
                printDiv();
                for (int i = 0; i < cnt; ++i)
                    cout << left << setw(9)  << simBuf[i].common
                                 << setw(10) << fixed << setprecision(4) << simBuf[i].jaccard
                                 << setw(8)  << simBuf[i].id
                                 << trunc(simBuf[i].title) << "\n";
                printDiv();
            }
        }

        // 3: BFS
        else if (ch == "3") {
            cout << "Enter paper ID or title keyword: ";
            string q; getline(cin, q);
            int pid = lookupPaper(q, papers);
            if (pid == -1) { cout << "Paper not found.\n"; continue; }

            cout << "Max hops (depth) [default 2]: ";
            string hs; getline(cin, hs);
            int hops = 2;
            try { if (!hs.empty()) hops = stoi(hs); } catch (...) {}

            cout << "\nBFS from: [" << pid << "] "
                 << trunc(papers.count(pid) ? papers.at(pid) : "?", 60) << "\n";
            cout << "Computing BFS up to " << hops << " hop(s)...\n";

            unordered_set<int> reachable = graph.bfsReachable(pid, hops);
            cout << "\nReachable papers (within " << hops
                 << " hop(s)): " << reachable.size() << "\n";

            int shown = 0;
            printDiv();
            for (int r : reachable) {
                cout << "  [" << r << "] "
                     << trunc(papers.count(r) ? papers.at(r) : "?", 65) << "\n";
                if (++shown >= 10) { cout << "  ... (showing first 10)\n"; break; }
            }
            printDiv();
        }

        // 4: Lookup
        else if (ch == "4") {
            cout << "Enter paper ID or title keyword: ";
            string q; getline(cin, q);
            int pid = lookupPaper(q, papers);
            if (pid == -1) { cout << "Paper not found.\n"; continue; }
            cout << "\nID:             " << pid << "\n";
            cout << "Title:          "
                 << (papers.count(pid) ? papers.at(pid) : "Unknown") << "\n";
            cout << "Cited by others (in-degree):  " << graph.inDegree(pid)  << "\n";
            cout << "Papers it cites (out-degree): " << graph.outDegree(pid) << "\n";
        }

        // 5: Stats
        else if (ch == "5") {
            int maxIn = 0, mostCited = -1;
            for (auto& [node, citedBy] : graph.radj) {
                int deg = static_cast<int>(citedBy.size());
                if (deg > maxIn) { maxIn = deg; mostCited = node; }
            }
            int E = graph.edgeCount();
            double avg = papers.empty() ? 0.0
                       : static_cast<double>(E) / static_cast<double>(papers.size());
            cout << "\nGraph Statistics\n";
            printDiv();
            cout << "  Nodes (papers loaded):       " << setw(10) << papers.size()       << "\n";
            cout << "  Nodes with out-edges:        " << setw(10) << graph.adj.size()     << "\n";
            cout << "  Nodes with in-edges:         " << setw(10) << graph.radj.size()    << "\n";
            cout << "  Total directed edges:        " << setw(10) << E                    << "\n";
            cout << "  Avg out-degree:              " << setw(10) << fixed << setprecision(2) << avg << "\n";
            cout << "  Max in-degree (times cited): " << setw(10) << maxIn                << "\n";
            if (mostCited != -1)
                cout << "  Most cited: [" << mostCited << "] "
                     << trunc(papers.count(mostCited) ? papers.at(mostCited) : "?") << "\n";
            printDiv();
        }

        else {
            cout << "Invalid choice.\n";
        }
    }
}


int main(int argc, char* argv[]) {
    string nodesPath = (argc > 1) ? argv[1] : "nodes.txt";
    string edgesPath = (argc > 2) ? argv[2] : "edges.txt";

    cout << "Loading nodes ... "; cout.flush();
    PaperMap papers = loadNodes(nodesPath);
    cout << papers.size() << " papers.\n";

    cout << "Loading edges into graph ... "; cout.flush();
    CitationGraph graph;
    loadEdges(edgesPath, graph);
    cout << graph.edgeCount() << " edges.\n";

    cout << "Building max-heap from in-degrees ... "; cout.flush();
    MaxHeap heap = buildMaxHeap(graph);
    cout << "Done.\n";

    runCLI(papers, graph, heap);
    return 0;
}

