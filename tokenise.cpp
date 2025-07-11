
#include "include/tokenise.hpp"
#include <string>
#include <sstream>
#include <string_view>
#include <regex>

#ifdef USE_OPENCL

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 */
tokeniser::tokeniser(int d, OpenCLContext& context) : d(d), ocl(context) {}

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 * @param vocSize vocabulary size
 */
tokeniser::tokeniser(int d, int d_val, OpenCLContext& context) : d(d), d_val(d_val), ocl(context) {}

/**
 * @brief constructor for tokeniser (use data set directly)
 * @param path2data path to folder with all dataset files
 */
tokeniser::tokeniser(const std::string& path2data, OpenCLContext& context) noexcept : path2data(path2data), ocl(context) 
{
    this->mappedEmbeddings = readMappedEmbeddings(path2data + "/_final_embeddings.csv");
    this->statOfTokens = readCorpusWordCount(path2data + "/_final_token_stats.csv");
    this->vocSize = this->statOfTokens.size() - 1;
    this->d = this->mappedEmbeddings.begin()->second.size() - 1;
}


#elif USE_CUDA || USE_CPU

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 */
tokeniser::tokeniser(int d) : d(d) {}

/**
 * @brief constructor for tokeniser (use dimensions directly)
 * @param d dimension for embedding
 * @param vocSize vocabulary size
 */
tokeniser::tokeniser(int d, int d_val) : d(d), d_val(d_val) {}

/**
 * @brief constructor for tokeniser (use data set directly)
 * @param path2data path to folder with all dataset files
 */
tokeniser::tokeniser(const std::string& path2data) noexcept : path2data(path2data)
{
    this->mappedEmbeddings = readMappedEmbeddings(path2data + "/_final_embeddings.csv");
    this->statOfTokens = readCorpusWordCount(path2data + "/_final_token_stats.csv");
    this->vocSize = this->statOfTokens.size() - 1;
    this->d = this->mappedEmbeddings.begin()->second.size() - 1;
}

#endif


/**
 * @brief copy constructor
 * @param toBeCopied constructor to be copied
 */
tokeniser::tokeniser(const tokeniser& toBeCopied) noexcept // Corrected parameter name and added const
    : d(toBeCopied.d),
      vocSize(toBeCopied.vocSize),
      path2data(toBeCopied.path2data),
      tokens(toBeCopied.tokens),
      embeddings(toBeCopied.embeddings),
      seeds(toBeCopied.seeds),
      deEmbeddings(toBeCopied.deEmbeddings),
      mappedEmbeddings(toBeCopied.mappedEmbeddings)
{
    #ifdef USE_OPENCL
        ocl = toBeCopied.ocl;
    #endif
}

/**
 * @brief move constructor
 * @param toBeMoved constructor to be moved
 */
tokeniser::tokeniser(tokeniser&& toBeMoved) noexcept
    : d(toBeMoved.d),
      vocSize(toBeMoved.vocSize),
      path2data(std::move(toBeMoved.path2data)),
      tokens(std::move(toBeMoved.tokens)),
      embeddings(std::move(toBeMoved.embeddings)),
      seeds(std::move(toBeMoved.seeds)),
      deEmbeddings(std::move(toBeMoved.deEmbeddings)),
      mappedEmbeddings(std::move(toBeMoved.mappedEmbeddings))
{
    #ifdef USE_OPENCL
        ocl = toBeMoved.ocl;
    #endif
}


// Helper function to allow splitting file_paths among producers
// This is not a member function, can be placed in the .cpp file before buildCorpusWordCountsParallel
std::vector<std::vector<std::string>> split_vector_for_producers(const std::vector<std::string>& files, 
    int num_producers)
{
    std::vector<std::vector<std::string>> splits(num_producers);
    size_t files_per_producer = files.size() / num_producers;
    size_t remainder = files.size() % num_producers;

    size_t current_file_idx = 0;
    for (int i = 0; i < num_producers; ++i) {
        size_t count = files_per_producer + (i < remainder ? 1 : 0);
        for (size_t j = 0; j < count; ++j) {
            splits[i].push_back(files[current_file_idx++]);
        }
    }
    return splits;
}