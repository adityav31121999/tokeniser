
#include "include/tokenise.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <cmath>

/**
 * @brief general formula for embedding
 * @param i token number
 * @param j specific position of embedding vector
 * @param seed seed for whole embedding
 */
float tokeniser::embeddingFormula(int i, int j, float seed)
{
    // This function implements the same logic as the embeddingFormulaLambda.
    // It's good practice to avoid duplication. We can call the static lambda here.
    // Note: `this->d` is used, which is the embedding dimension.
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
    // Use std::generate with a stateful lambda to populate the vector.
    // The lambda captures a counter 'j' to represent the position in the embedding.
    int j = 0;
    std::generate(embed.begin(), embed.end(), [&j, i, d_val = d, seed] {
        // Call the static lambda from the header.
        return tokeniser::embeddingFormulaLambda(i, j++, d_val, seed);
    });

    return embed;
}