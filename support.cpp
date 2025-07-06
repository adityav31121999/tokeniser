// support functions
#include <iostream>
#include <fstream>
#include <random>       // For std::mt19937 and std::uniform_real_distribution
#include <iomanip>      // For std::fixed and std::setprecision
#include <vector>
#include <algorithm>    // For std::sort
#include <cmath>        // For std::abs
#include "include/tokenise.hpp"


/**
 * @brief multiplicative inverse of a vector
 * @param vec vector input
 * @return inverse of vector
 */
std::vector<float> vectorInverse(const std::vector<float> &vec)
{
    std::vector<float> inverse(vec.size());
    float magnitudeOfVec = 0.0f;
    for (float val : vec) {
        magnitudeOfVec += val * val;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        inverse[i] = vec[i]/magnitudeOfVec;
    }
    return inverse;
}

/**
 * @brief Generates a random seed for each token in the vocabulary.
 * The seeds are uniformly distributed within the specified range [r1, r2].
 * @param r1 Lower bound for the random seed generation.
 * @param r2 Upper bound for the random seed generation.
 */
void tokeniser::seedsForEmbedding(float r1, float r2) {
    if (this->tokens.empty()) {
        std::cerr << "Warning: Cannot generate seeds for an empty vocabulary." << std::endl;
        return;
    }
    
    // Modern C++ way to generate random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(r1, r2);

    this->seeds.resize(this->vocSize);
    for (int i = 0; i < this->vocSize; ++i) {
        this->seeds[i] = distrib(gen);
    }
}


/**
 * @brief Generates random seeds and computes embeddings for the current vocabulary.
 * This function populates the internal embedding vectors based on the tokens
 * currently stored in the class. It uses either the CPU, CUDA, or OpenCL
 * implementation to calculate the embeddings and their inverses. Finally,
 * it saves the token-embedding pairs to a specified CSV file.
 * @param outputPath The file path where the token-embedding CSV should be saved.
 * @param r1 The lower bound for the random seed generation.
 * @param r2 The upper bound for the random seed generation.
 * @throws std::runtime_error if the output file cannot be opened.
 */
void tokeniser::generateAndSaveEmbeddings(const std::string& outputPath, float r1, float r2) {
    if (this->tokens.empty()) {
        throw std::runtime_error("Error: Vocabulary is not trained. Cannot generate embeddings.");
    }
    this->vocSize = this->tokens.size();

    // Create a temporary, alphabetically sorted list of tokens for generating
    // consistent seeds and for saving to the CSV file.
    // This leaves the internal `this->tokens` (sorted by length) untouched.
    std::cout << "-> Creating a temporary, lexicographically sorted token list for saving..." << std::endl;
    std::vector<std::string> sorted_tokens_for_saving = this->tokens;
    std::sort(sorted_tokens_for_saving.begin(), sorted_tokens_for_saving.end());

    // Create a mapping from the alphabetically sorted tokens back to their original
    // positions in the length-sorted list. This is crucial for calculating embeddings correctly.
    std::map<std::string, int> token_to_original_index;
    for(int i = 0; i < this->vocSize; ++i) {
        token_to_original_index[this->tokens[i]] = i;
    }

    // 1. Generate seeds for the vocabulary
    std::cout << "-> Generating random seeds for " << this->vocSize << " tokens..." << std::endl;
    this->seedsForEmbedding(r1, r2);

    // 2. Calculate embeddings
    std::cout << "-> Calculating " << this->d << "-dimensional embeddings..." << std::endl;
    
    // Resize vectors to hold the data
    this->embeddings.resize(this->vocSize, std::vector<float>(this->d));
    this->deEmbeddings.resize(this->vocSize, std::vector<float>(this->d));
    
    // We will calculate embeddings based on the ORIGINAL length-sorted order
    // to maintain consistency with any potential future use.
    #ifdef USE_CUDA
        // Call the CUDA kernel wrapper
        cuEmbeddingFormula(this->embeddings, this->seeds, this->d, this->vocSize);
        cuVectorInverse(this->deEmbeddings, this->embeddings, this->d, this->vocSize);
    #elif USE_OPENCL
        // Call the OpenCL kernel wrapper
        clEmbeddingFormula(this->ocl, this->embeddings, this->seeds, this->d, this->vocSize);
        clVectorInverse(this->ocl, this->deEmbeddings, this->embeddings, this->d, this->vocSize);
    #else
        // Fallback to CPU implementation
        for (int i = 0; i < this->vocSize; ++i) {
            // Note: `i` here corresponds to the index in the original `this->tokens` list
            for (int j = 0; j < this->d; ++j) {
                this->embeddings[i][j] = embeddingFormulaLambda(i, j, this->d_val, this->seeds[i]);
            }
        }
    #endif

    std::cout << "-> Embedding generation complete." << std::endl;

    // 3. Save to CSV file using the alphabetically sorted list
    std::cout << "-> Saving tokens and embeddings to: " << outputPath << std::endl;
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open output file: " + outputPath);
    }
    
    // Set precision for floating point numbers
    outFile << std::fixed << std::setprecision(8);

    // --- MODIFICATION: Write data using the sorted list ---
    // Iterate through the alphabetically sorted tokens
    for (const auto& token : sorted_tokens_for_saving) {
        // Find the original index to get the correct embedding
        int original_index = token_to_original_index.at(token);
        
        // Handle tokens that might contain commas by quoting them
        outFile << "\"" << token << "\"";
        for (int j = 0; j < this->d; ++j) {
            outFile << "," << this->embeddings[original_index][j];
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "-> Successfully saved file." << std::endl;
}