#include "include/tokenise.hpp"
#include <iostream>
#include <algorithm>
#include <future>
#include <fstream>

/**
 * @brief Calculates the frequency of all adjacent pairs of symbols.
 * This function is a core component of the Byte Pair Encoding (BPE) algorithm.
 * It iterates through all words and their current token splits to count how
 * many times each adjacent pair of tokens appears.
 * @param word_counts A map from each word in the corpus to its frequency.
 * @param splits A map from each word to its current representation as a vector of subword tokens.
 * @return A map where keys are pairs of adjacent tokens and values are their total frequency.
 */
std::map<std::pair<std::string, std::string>, int> get_pair_stats( const std::map<std::string, int>& word_counts,
    const std::map<std::string, std::vector<std::string>>& splits) 
{
    std::map<std::pair<std::string, std::string>, int> pair_freqs;
    for (const auto& pair : word_counts) {
        const std::string& word = pair.first;
        const int count = pair.second;
        const std::vector<std::string>& symbols = splits.at(word);
        for (size_t i = 0; i < symbols.size() - 1; ++i) {
            pair_freqs[{symbols[i], symbols[i + 1]}] += count;
        }
    }
    return pair_freqs;
}


/**
 * @brief Calculates the frequency of all adjacent pairs of symbols.
 * Splits the workload (unique words) across multiple threads for concurrent processing.
 * @param word_counts Map from each word to its frequency.
 * @param splits Map from each word to its current token representation.
 * @param num_threads Number of threads to use.
 * @return A map of pair frequencies, aggregated from all threads.
 */
std::map<std::pair<std::string, std::string>, int> parallel_get_pair_stats(const std::map<std::string, int>& word_counts,
    const std::map<std::string, std::vector<std::string>>& splits, int num_threads)
{
    std::vector<std::string> words;
    words.reserve(word_counts.size());
    for(const auto& pair : word_counts) {
        words.push_back(pair.first);
    }

    std::vector<std::future<std::map<std::pair<std::string, std::string>, int>>> futures;
    int chunk_size = (words.size() + num_threads - 1) / num_threads; // Ceiling division

    for (int i = 0; i < num_threads; ++i) {
        int start_idx = i * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, (int)words.size());
        if (start_idx >= end_idx) continue;

        futures.push_back(std::async(std::launch::async, 
            [start_idx, end_idx, &words, &word_counts, &splits] {
                std::map<std::pair<std::string, std::string>, int> local_pair_freqs;
                for (int j = start_idx; j < end_idx; ++j) {
                    const std::string& word = words[j];
                    const int count = word_counts.at(word);
                    const std::vector<std::string>& symbols = splits.at(word);
                    if (symbols.size() < 2) continue;
                    for (size_t k = 0; k < symbols.size() - 1; ++k) {
                        local_pair_freqs[{symbols[k], symbols[k + 1]}] += count;
                    }
                }
                return local_pair_freqs;
            }
        ));
    }

    // Aggregate results from all threads
    std::map<std::pair<std::string, std::string>, int> total_pair_freqs;
    for (auto& future : futures) {
        auto local_map = future.get();
        for (const auto& pair : local_map) {
            total_pair_freqs[pair.first] += pair.second;
        }
    }
    return total_pair_freqs;
}


/**
 * @brief Calculates the frequency of each final token based on the trained vocabulary.
 * This function iterates through the original pre-tokenized corpus, splits words into
 * subwords using the learned BPE vocabulary, and counts the occurrences of each final
 * token. The results are stored in the `statOfEmbeddings` member variable. If an
 * output path is provided, it saves the statistics to a two-column CSV file
 * ('token', 'repetitions').
 * @param pre_tokens A vector of all words and punctuation from the training corpus.
 * @param outputPath The path to the CSV file for saving token statistics. If empty, no file is saved.
 */
void tokeniser::calculateTokenStats(const std::vector<std::string>& pre_tokens, const std::string& outputPath) {
    this->statOfEmbeddings.clear();

    // Helper lambda to identify words that were processed by BPE.
    // This must be consistent with the logic in `groupCommonTokens`.
    auto is_word_for_bpe = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        // A simple check: does it start with a letter?
        return std::isalpha(static_cast<unsigned char>(s[0]));
    };

    for (const auto& pre_token : pre_tokens) {
        if (is_word_for_bpe(pre_token)) {
            std::vector<std::string> subwords;
            // `splitWord` tokenizes a single word using the final vocabulary.
            this->splitWord(pre_token, subwords);
            for (const auto& subword : subwords) {
                this->statOfEmbeddings[subword]++;
            }
        }
        else {
            // It's punctuation or another symbol, which is treated as an atomic token.
            this->statOfEmbeddings[pre_token]++;
        }
    }

    if (!outputPath.empty()) {
        std::cout << "-> Saving token statistics to: " << outputPath << std::endl;
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            // Warn the user but don't stop the program, as saving stats might be optional.
            std::cerr << "Warning: Could not open file to save token stats: " << outputPath << std::endl;
            return;
        }

        // Write header
        outFile << "token,repetitions\n";

        // Write data
        for (const auto& pair : this->statOfEmbeddings) {
            // Handle tokens that might contain commas by quoting them
            outFile << "\"" << pair.first << "\"," << pair.second << "\n";
        }

        outFile.close();
        std::cout << "-> Successfully saved statistics file." << std::endl;
    }
}
