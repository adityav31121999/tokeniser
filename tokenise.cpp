
#include "include/tokenise.hpp"

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
tokeniser::tokeniser(const std::string& path2data) noexcept : path2data(path2data) {}

/**
 * @brief constructor for tokeniser
 * @param d dimension for embedding
 * @param vocSize vocabulary size
 * @param path2data path to folder with all dataset files
 */ 
tokeniser::tokeniser(const std::string& path2data, int d, int d_val) noexcept
    : path2data(path2data), d(d), d_val(d_val) {}

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
{}

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
{}
