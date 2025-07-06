#include "include/tokenise.hpp"


/**
 * @brief (HELPER) Merges two maps into one.
 * The contents of 'source' are moved into 'destination'.
 */
void merge_maps(std::map<std::string, int>& destination, std::map<std::string, int>& source) {
    if (source.size() > destination.size()) {
        source.swap(destination); // Ensure we iterate over the smaller map
    }
    for (const auto& pair : source) {
        destination[pair.first] += pair.second;
    }
}


/**
 * @brief Merges a specified pair of tokens into a new single token across all word splits.
 * This function performs the "merge" step of the BPE algorithm by replacing all
 * occurrences of the most frequent pair with a new combined token.
 * @param best_pair The pair of tokens to be merged (e.g., {"t", "h"}).
 * @param splits The map of word splits to be updated. This is modified in place.
 */
void merge_pair(const std::pair<std::string, std::string>& best_pair,
                std::map<std::string, std::vector<std::string>>& splits) 
{
    std::string new_token = best_pair.first + best_pair.second;
    for (auto& pair : splits) {
        std::vector<std::string>& symbols = pair.second;
        std::vector<std::string> new_symbols;
        size_t i = 0;
        while (i < symbols.size()) {
            if (i < symbols.size() - 1 && symbols[i] == best_pair.first && symbols[i+1] == best_pair.second) {
                new_symbols.push_back(new_token);
                i += 2;
            } 
            else {
                new_symbols.push_back(symbols[i]);
                i += 1;
            }
        }
        symbols = new_symbols;
    }
}


/**
 * @brief Merges a specified pair of tokens into a new single token across all word splits.
 * This is highly parallelizable as each word's split is independent.
 * @param best_pair The pair of tokens to be merged.
 * @param splits The map of word splits to be updated in place.
 * @param num_threads Number of threads to use.
 */
void parallel_merge_pair(const std::pair<std::string, std::string>& best_pair,
    std::map<std::string, std::vector<std::string>>& splits, int num_threads)
{
    std::vector<std::string> words_to_process;
    words_to_process.reserve(splits.size());
    for(const auto& pair : splits) {
        words_to_process.push_back(pair.first);
    }

    std::vector<std::thread> threads;
    int chunk_size = (words_to_process.size() + num_threads - 1) / num_threads;
    const std::string new_token = best_pair.first + best_pair.second;

    for (int i = 0; i < num_threads; ++i) {
        int start_idx = i * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, (int)words_to_process.size());
        if (start_idx >= end_idx) continue;

        threads.emplace_back([start_idx, end_idx, &words_to_process, &splits, &best_pair, &new_token] {
            for (int j = start_idx; j < end_idx; ++j) {
                const std::string& word = words_to_process[j];
                std::vector<std::string>& symbols = splits.at(word);
                if (symbols.size() < 2) continue;

                std::vector<std::string> new_symbols;
                new_symbols.reserve(symbols.size()); // Pre-allocate to avoid reallocations
                size_t k = 0;
                while (k < symbols.size()) {
                    if (k < symbols.size() - 1 && symbols[k] == best_pair.first && symbols[k + 1] == best_pair.second) {
                        new_symbols.push_back(new_token);
                        k += 2;
                    } else {
                        new_symbols.push_back(symbols[k]);
                        k += 1;
                    }
                }
                symbols = std::move(new_symbols); // Efficiently replace the old vector
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

