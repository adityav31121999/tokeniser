#ifndef TOKENISE_HPP
#define TOKENISE_HPP 1

#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <cmath>
#include "clcontext.hpp"

// struct for token related data
struct token {
    std::string token;
    float seed;
    std::vector<float> embedding;
    std::vector<float> deEmbedding;
};

/**
 * Class to tokenise dataset into subwords and embeddings. The embeddings are of 
 * d dimension with all the values of type float.
 */
class tokeniser {
private:
    int d;          // embedding dimension
    int vocSize;    // vocabulary size
    int d_val;      // divisor (makes it look like repeatation)

    std::string path2data;                          // path to dataset
    std::vector<std::string> tokens;                // all possible tokens
    std::vector<float> seeds;                       // seeds for all tokens
    std::vector<std::vector<float>> embeddings;     // vector for each token of dimension d
    std::vector<std::vector<float>> deEmbeddings;   // inverse of each token of dimension d
    // map of tokens and their embeddings
    std::map<std::string, std::vector<float>> mappedEmbeddings;
    std::map<std::string, int> statOfEmbeddings;    // hold number of uses of each token
    FILE* csvTokensAndEmbeddings = nullptr;         // hold token and its respective embedding
    FILE* csvTokensAndSeeds = nullptr;              // hold all tokens and seeds

public:

    int num_threads;                                // number of threads

// constructors
    // default constructors
    tokeniser() = default;
    tokeniser(int d);
    tokeniser(int d, int d_val);
    explicit tokeniser(const std::string& path2data) noexcept;
    explicit tokeniser(const std::string& path2data, int d, int d_val) noexcept;
    tokeniser(const tokeniser& toBeCopied) noexcept;
    tokeniser(tokeniser&& toBeMoved) noexcept;

// programs for setting values
    void setEmbeddingDimension(int d);
    void setDval(int d_val);
    void setVocabularySize(int vocSize);
    void setNumThreads();
    void setEmbedding(const std::string& token, std::vector<float> embedding);

    // Getters for read-only access to internal state
    int getEmbeddingDimension() const { return d; }
    int getDval() const { return d_val; }
    int getVocabularySize() const { return vocSize; }
    const std::map<std::string, int>& getTokenStats() const { return statOfEmbeddings; }
    const std::vector<std::string>& getTokens() const { return tokens; }
    const std::map<std::string, std::vector<float>>& getMappedEmbeddings() const { return mappedEmbeddings; }
    const std::vector<float>& getSeeds() const { return seeds; }
    const std::vector<std::vector<float>>& getEmbeddings() const { return embeddings; }
    const std::vector<std::vector<float>>& getDeEmbeddings() const { return deEmbeddings; }
    std::vector<float> getEmbeddingForToken(const std::string& token) const;
    std::vector<float> getEmbeddingForToken(int index) const { return embeddings[index]; };

    void splitWord(const std::string& word, std::vector<std::string>& subwords) const;
    void splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords) const;
    void splitWordsFromTxt(const std::string& path2txt, std::vector<std::string>& words);
    void splitWordsFromTxtParallel(const std::string& path2txt, std::vector<std::string>& words);
    void splitWordsFromCSV(const std::string& path2csv, std::vector<std::string>& tokens);
    // void arrangeTokens(std::vector<std::string>& tokens);
    void groupCommonTokens(std::vector<std::string>& words, int num_merges, std::vector<std::string>& final_vocab);
    void groupCommonTokensParallel(std::vector<std::string>& words, int num_merges, std::vector<std::string>& final_vocab);
    void calculateTokenStats(const std::vector<std::string>& pre_tokens, const std::string& outputPath);
    std::vector<std::string> tokeniseFile(const std::string& filePath) const;

    /**
     * a generic lambda that implements the mathematical formula:
     * --> f(i, j, seed) = (i * j + 1) * C * (seed^[j%d])
     * where: C = 0.01, x = seed, and  d is the embedding dimension.
     */
    static constexpr auto embeddingFormulaLambda = [](int i, int j, int d_val, auto seed_val)
    {
        // (i * j + 1) * C
        // C = 0.01
        float intermediate_val = static_cast<float>(j + 1) * 0.01f / j;
        int exponent = j % d_val;
        // (seed^[j%d])
        intermediate_val *= std::pow(seed_val, exponent+1);
        return intermediate_val;
    };

    void seedsForEmbedding(float r1, float r2);
    float embeddingFormula(int i, int j, float seed);
    std::vector<float> embeddingFormula(int i, float seed);
    void generateAndSaveEmbeddings(const std::string& outputPath, float r1, float r2);

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


std::map<std::pair<std::string, std::string>, int> get_pair_stats(const std::map<std::string, int>& word_counts,
    const std::map<std::string, std::vector<std::string>>& splits);
void merge_pair(const std::pair<std::string, std::string>& best_pair, std::map<std::string, std::vector<std::string>>& splits);
std::map<std::pair<std::string, std::string>, int> parallel_get_pair_stats(const std::map<std::string, int>& word_counts,
    const std::map<std::string, std::vector<std::string>>& splits, int num_threads);
void parallel_merge_pair(const std::pair<std::string, std::string>& best_pair, std::map<std::string, std::vector<std::string>>& splits,
    int num_threads);


std::vector<float> vectorInverse(const std::vector<float>& vec);

#ifdef USE_CUDA
// kernel for embedding calculation
// kernel for vector inverse calculation

#endif

#endif