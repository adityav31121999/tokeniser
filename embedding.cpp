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
 * @brief general formula for embedding
 * @param i token number
 * @param j specific position of embedding vector
 * @param seed seed for whole embedding
 */
float tokeniser::embeddingFormula(int i, int j, float seed) {
    return tokeniser::embeddingFormulaLambda(i, j, d_val, seed);
}


/**
 * @brief generates an embedding vector for a given token number
 * @param i token number
 * @param seed seed for whole embedding
 */
std::vector<float> tokeniser::embeddingFormula(int i, float seed)
{
    // Initialize a vector of size 'd' to hold the embedding.
    std::vector<float> embed(d, 0.0f);
    int j = 0;
    std::generate(embed.begin(), embed.end(), [&j, i, d_val = d, seed] {
        // Call the static lambda from the header.
        return tokeniser::embeddingFormulaLambda(i, j++, d_val, seed);
    });
    return embed;
}


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
void tokeniser::seedsForEmbedding(float r1, float r2, const std::string& seedCSVpath) {
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

    std::cout << "-> Seeds created." << std::endl;
    std::cout << "-> Saving tokens and embeddings to: " << seedCSVpath << std::endl;
    std::ofstream outFile(seedCSVpath);

    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open output file: " + seedCSVpath);
    }
    outFile << std::fixed << std::setprecision(8);

    // Iterate through tokens
    int i = 0;
    outFile << "token" <<  "," << "seed" << "\n";
    for (int i = 0; i < this->vocSize; i++) {
        outFile << tokens[i] <<  "," << this->seeds[i] << "\n";
    }
    outFile.close();
    std::cout << "-> Successfully saved file." << std::endl;
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
void tokeniser::generateAndSaveEmbeddings(const std::string& embeddingCSVpath, const std::string& seedCSVpath, float r1, float r2) {
    if (this->tokens.empty()) {
        throw std::runtime_error("Error: Vocabulary is not trained. Cannot generate embeddings.");
    }
    this->vocSize = this->tokens.size();

    std::cout << "-> Creating a temporary, lexicographically sorted token list for saving..." << std::endl;
    std::vector<std::string> sorted_tokens_for_saving = this->tokens;
    // std::sort(sorted_tokens_for_saving.begin(), sorted_tokens_for_saving.end());
    std::unordered_map<std::string, int> token_to_original_index;

    for(int i = 0; i < this->vocSize; ++i) {
        token_to_original_index[this->tokens[i]] = i;
    }

    // Generate seeds for the vocabulary
    std::cout << "-> Generating random seeds for " << this->vocSize << " tokens..." << std::endl;
    this->seedsForEmbedding(r1, r2, seedCSVpath);
    std::cout << "-> Calculating " << this->d << "-dimensional embeddings..." << std::endl;
    
    // Resize vectors to hold the data
    this->embeddings.resize(this->vocSize, std::vector<float>(this->d));
    this->deEmbeddings.resize(this->vocSize, std::vector<float>(this->d));
    
    #ifdef USE_CUDA
        // Call the CUDA kernel wrapper
        cuEmbeddingFormula(this->embeddings, this->seeds, this->d, this->vocSize);
        // cuVectorInverse(this->deEmbeddings, this->embeddings, this->d, this->vocSize);
    #elif USE_OPENCL
        // Call the OpenCL kernel wrapper
        clEmbeddingFormula(this->ocl, this->embeddings, this->seeds, this->d, this->vocSize);
        // clVectorInverse(this->ocl, this->deEmbeddings, this->embeddings, this->d, this->vocSize);
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
    std::cout << "-> Saving tokens and embeddings to: " << embeddingCSVpath << std::endl;
    std::ofstream outFile(embeddingCSVpath);

    if (!outFile.is_open()) {
        throw std::runtime_error("Error: Could not open output file: " + embeddingCSVpath);
    }
    outFile << std::fixed << std::setprecision(8);

    // Iterate through the alphabetically sorted tokens
    for (const auto& token : sorted_tokens_for_saving) {
        // Find the original index to get the correct embedding
        int original_index = token_to_original_index.at(token);
        outFile << "\"" << token << "\"";

        for (int j = 0; j < this->d; ++j) {
            outFile << "," << this->embeddings[original_index][j];
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "-> Successfully saved file." << std::endl;
}
