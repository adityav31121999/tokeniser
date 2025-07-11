// header file
#ifndef TOKENISE_HPP
#define TOKENISE_HPP 1

#include "clcontext.hpp"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <cmath>
#include <queue>                 // For std::queue
#include <mutex>                 // For std::mutex, std::lock_guard
#include <condition_variable>    // For std::condition_variable
#include <unordered_map>         // For the much faster hash map
#include <regex>
#include <algorithm>
#include <future>

// for fast multithreading operations
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};


/**
 * @brief A struct to hold shared progress data for logging.
 * Protected by a mutex.
 */
struct ProgressData {
    std::mutex mtx;
    std::condition_variable cv;
    long long total_bytes = 0;
    long long bytes_read = 0;
    size_t files_completed_count = 0;
    std::string last_file_completed;
    int merges_completed = 0;
    int total_merges = 0;
    std::chrono::steady_clock::time_point start_time;
};


/**
 * Class to tokenise dataset into subwords and embeddings. The embeddings are of 
 * d dimension with all the values of type float.
 */
class tokeniser {
private:
    int d;          // embedding dimension
    int vocSize;    // vocabulary size (number of merges performed while BPE algorithm)
    int d_val;      // divisor (makes it look like repeatation)

    std::string path2data;                          // path to dataset
    std::vector<std::string> tokens;                // all possible tokens
    std::vector<float> seeds;                       // seeds for all tokens (seeds.csv)
    std::vector<std::vector<float>> embeddings;     // vector for each token of dimension d
    std::vector<std::vector<float>> deEmbeddings;   // inverse of each token of dimension d
    // map of tokens and their embeddings (embeddings.csv)
    std::unordered_map<std::string, std::vector<float>> mappedEmbeddings;
    std::unordered_map<std::string, int> corpusWordCount;   // NEW (or similar if it's not a member)
    std::unordered_map<std::string, int> statOfTokens;      // hold tokens and their stats (unique_tokens.csv)

public:

    int num_threads;                        // number of threads
    int totalCorpusWordCount;               // corpus word count
    ProgressData bpe_progress;              // track progress throughout for tokenisation

// constructors
    // default constructors
#ifdef USE_OPENCL
    tokeniser(int d, OpenCLContext& context);
    tokeniser(int d, int d_val, OpenCLContext& context);
    explicit tokeniser(const std::string& path2data, OpenCLContext& context) noexcept;
#elif USE_CUDA || USE_CPU
    tokeniser() = default;
    tokeniser(int d);
    tokeniser(int d, int d_val);
    explicit tokeniser(const std::string& path2data) noexcept;
#endif

    tokeniser(const tokeniser& toBeCopied) noexcept;        // copy constructor
    tokeniser(tokeniser&& toBeMoved) noexcept;              // move constructor

    /**
     * a generic lambda that implements the mathematical formula:
     * --> f(i, j, seed) = (j + 1) * C * (seed^[j%d + 1]) / (j%d + 1)
     * where: C = 0.01, x = seed, and  d is the embedding dimension.
     */
    static constexpr auto embeddingFormulaLambda = [](int i, int j, int d_val, auto seed_val)
    {
        // (i * j + 1) * C
        // C = 0.01
        int exponent = (j % d_val) + 1;
        float intermediate_val = static_cast<float>(j + 1) * 0.01f / exponent;
        // (seed^[j%d])
        intermediate_val *= std::pow(seed_val, exponent);
        return intermediate_val;
    };

    // programs for setting values
    void setEmbeddingDimension(int d);
    void setDval(int d_val);
    void setVocabularySize(int vocSize);
    void setNumThreads();
    void setEmbedding(const std::string& token, std::vector<float> embedding);
    void readFromFiles(const std::string& path2ClassDataFolder);

    // Getters for read-only access to internal state
    int getEmbeddingDimension() const { return d; }
    int getDval() const { return d_val; }
    int getVocabularySize() const { return vocSize; }
    const std::unordered_map<std::string, int>& getTokenStats() const { return statOfTokens; }
    const std::vector<std::string>& getTokens() const { return tokens; }
    const std::unordered_map<std::string, std::vector<float>>& getMappedEmbeddings() const { return mappedEmbeddings; }
    const std::vector<float>& getSeeds() const { return seeds; }
    const std::vector<std::vector<float>>& getEmbeddings() const { return embeddings; }
    const std::vector<std::vector<float>>& getDeEmbeddings() const { return deEmbeddings; }
    std::vector<float> getEmbeddingForToken(int index) const { return embeddings[index]; };
    std::vector<float> getEmbeddingForToken(const std::string& token) const;

    void splitWord(const std::string& word, std::vector<std::string>& subwords) const;
    void splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords) const;
    void buildCorpusWordCounts(const std::vector<std::string>& file_paths, std::unordered_map<std::string, int>& corpus_word_counts);
    void groupCommonTokens(const std::unordered_map<std::string, int>& corpus_word_counts, int num_merges, std::vector<std::string>& final_vocab);
    void learn_vocabulary_from_word_counts(const std::unordered_map<std::string, int>& corpus_word_counts, int num_merges, std::vector<std::string>& final_vocab);
    void saveUniqueTokensToCSV(const std::unordered_map<std::string, int>& corpus_word_counts, const std::string& outputPath);
    void calculateTokenStatsFromCounts(const std::unordered_map<std::string, int>& corpus_word_counts, const std::string& outputPath);
    void calculateTokenStats(const std::vector<std::string>& pre_tokens, const std::string& outputPath);

    void seedsForEmbedding(float r1, float r2,const std::string& seedCSVpath);
    float embeddingFormula(int i, int j, float seed);
    std::vector<float> embeddingFormula(int i, float seed);
    void generateAndSaveEmbeddings(const std::string& outputPath, const std::string& seedCSVpath, float r1, float r2);

    #ifdef USE_CUDA
        void cuEmbeddingFormula(std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds, int& d, int& vocSize);
        void cuVectorInverse(std::vector<std::vector<float>>& deEmbedding, const std::vector<std::vector<float>>& embedding, int& d, int& vocSize);
    #elif USE_OPENCL
        OpenCLContext ocl;
        void clEmbeddingFormula(OpenCLContext& ocl, std::vector<std::vector<float>>& embedding, const std::vector<float>& seeds, int& d, int& vocSize);
        void clVectorInverse(OpenCLContext& ocl, std::vector<std::vector<float>>& deEmbedding, const std::vector<std::vector<float>>& embedding, int& d, int& vocSize);
    #endif

    ~tokeniser() = default;
};

/**
 * @brief A thread-safe queue designed for producer-consumer patterns.
 * @tparam T The type of elements to store in the queue.
 */
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;

public:
    /**
     * @brief Pushes a new item onto the queue and notifies a waiting consumer.
     */
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    /**
     * @brief Waits for an item and pops it from the queue.
     * @param item The reference to store the popped item in.
     * @return `false` if the queue is closed and empty, `true` otherwise.
     */
    bool wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });

        if (queue_.empty() && done_) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    /**
     * @brief Signals to all consumers that production is complete.
     */
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_all(); // Wake up all waiting threads so they can exit.
    }
};


std::vector<std::string_view> pre_split_word(std::string_view word);
std::unordered_map<std::pair<std::string, std::string>, int, PairHash> get_pair_stats(const std::unordered_map<std::string, int>& word_counts, const std::unordered_map<std::string, std::vector<std::string>>& splits);
void merge_pair(const std::pair<std::string, std::string>& best_pair, std::unordered_map<std::string, std::vector<std::string>>& splits, int num_threads);
std::unordered_map<std::string, int> mergeTwoMaps(std::unordered_map<std::string, int> map1, std::unordered_map<std::string, int> map2);
std::future<std::unordered_map<std::string, int>> merge_maps(std::vector<std::future<std::unordered_map<std::string, int>>>& futures,size_t start_idx, size_t end_idx);
std::vector<float> vectorInverse(const std::vector<float>& vec);
std::vector<std::string> pre_tokenize_word_by_corpus_freq(const std::string& word, const std::unordered_map<std::string, int>& corpus_word_counts);

std::string trim(const std::string& str);
std::string removeQuotes(const std::string& str);
bool isHeaderLine(const std::string& line);
std::vector<std::string> readSingleColumnCsv(const std::string& filename);
std::vector<std::string> readSpecificColumnFromCsv(const std::string& filename, int targetColumnIndex);
std::vector<std::vector<std::string>> readCsvTo2DVector(const std::string& filename);
std::unordered_map<std::string, int> readUnorderedMap(const std::string& filename);
std::unordered_map<std::string, std::vector<float>> readMappedEmbeddings(const std::string& filename);

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// kernel for embedding calculation
__global__ void embeddingFormulaBatchKernel(float* all_embeddings, const float* all_seeds, 
            const int N, const int d);
// kernel for vector inverse calculation
__global__ void batchedVectorInverseKernel(float* output, const float* input, int N, int d);
#endif

#endif