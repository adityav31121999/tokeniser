#include "include/tokenise.hpp"
#include <iostream>
#include <algorithm>
#include <future>
#include <fstream>

/**
 * @brief Calculates final sub-token statistics from a pre-computed map of word counts
 *        and saves them to a CSV file, sorted alphanumerically by token. This version 
 *        is orders of magnitude faster than iterating over all tokens in the corpus.
 * @param corpus_word_counts Map of unique words and their frequencies.
 * @param outputPath Path to save the statistics CSV file.
 */
void tokeniser::calculateTokenStatsFromCounts(const std::unordered_map<std::string, int>& corpus_word_counts, 
    const std::string& outputPath) 
{
    this->statOfTokens.clear(); 

    auto is_word_for_bpe = [](const std::string& s) -> bool {
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0]));
    };

    std::cout << "Calculating final token statistics from " << corpus_word_counts.size() << " unique raw tokens..." << std::endl;

    const int num_threads = this->num_threads > 0 ? this->num_threads : 1;
    std::vector<std::future<std::unordered_map<std::string, int>>> futures;

    auto it = corpus_word_counts.begin();
    size_t total_items = corpus_word_counts.size();
    size_t chunk_size = (total_items + num_threads - 1) / num_threads; // Ceiling division

    for (int t = 0; t < num_threads; ++t) {
        auto chunk_start = it;
        // Advance iterator, ensuring we don't go past the end.
        std::advance(it, std::min<int>(chunk_size, (size_t)std::distance(it, corpus_word_counts.end())));
        auto chunk_end = it;

        if (chunk_start == chunk_end) {
            // No more work for this thread. This happens if total_items < num_threads
            continue; 
        }

        // Launch an asynchronous task for each chunk
        futures.push_back(std::async(std::launch::async, 
            [=, this]() -> std::unordered_map<std::string, int> { // Capture 'this' pointer for splitWord
            std::unordered_map<std::string, int> local_stats;
            
            for (auto current_it = chunk_start; current_it != chunk_end; ++current_it) {
                const std::string& pre_token = current_it->first;
                const int count = current_it->second;

                if (is_word_for_bpe(pre_token)) {
                    std::vector<std::string> subwords;
                    // Ensure this->splitWord is const-correct and thread-safe (read-only access to this->tokens)
                    this->splitWord(pre_token, subwords); 
                    for (const auto& subword : subwords) {
                        local_stats[subword] += count;
                    }
                } else {
                    local_stats[pre_token] += count;
                }
            }
            return local_stats;
        }));
    }

    // Aggregate results from all threads into the main statOfTokens (unordered_map)
    std::cout << "Aggregating parallel statistics: ";
    int merge_count = 0;
    for (auto& f : futures) {
        merge_count++;
        std::cout << merge_count << " ";
        auto local_map = f.get(); // Retrieve the unordered_map from the future
        for (const auto& pair : local_map) {
            this->statOfTokens[pair.first] += pair.second; // Merge into global unordered_map
        }
    }
    std::cout << std::endl;
    std::cout << "Calculation complete. Found " << this->statOfTokens.size() << " final BPE tokens." << std::endl;

    if (!outputPath.empty()) {
        std::cout << "-> Sorting and saving token statistics to: " << outputPath << std::endl;
        
        // 1. Copy the unordered_map contents to a vector of pairs for sorting.
        std::vector<std::pair<std::string, int>> sorted_stats(
            this->statOfTokens.begin(),
            this->statOfTokens.end()
        );

        // 2. Sort the vector.
        std::sort(sorted_stats.begin(), sorted_stats.end(), 
            [](const auto& a, const auto& b) {
                return a.first < b.first; // Explicitly sort by token string (the key)
            }
        );

        // 3. Write the sorted vector to the file.
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            std::cerr << "Warning: Could not open file to save token stats: " << outputPath << std::endl;
            return;
        }
        outFile << "token,repetitions\n";
        for (const auto& pair : sorted_stats) {
            const std::string& token = pair.first;
            // Handle tokens that might contain commas or quotes (CSV escaping)
            std::string escaped_token;
            bool needs_quotes = false;
            for (char c : token) {
                if (c == '"') {
                    escaped_token += "\"\""; // Double quotes
                    needs_quotes = true;
                }
                else if (c == ',') {
                    needs_quotes = true;
                    escaped_token += c;
                }
                else {
                    escaped_token += c;
                }
            }
            if (needs_quotes) {
                outFile << "\"" << escaped_token << "\"," << pair.second << "\n";
            } else {
                outFile << escaped_token << "," << pair.second << "\n";
            }
        }
        outFile.close();
        std::cout << "-> Successfully saved sorted statistics file." << std::endl;
    }
    else {
        std::cout << "\nOutput path is empty. Skipped saving statistics file." << std::endl;
    }
}


/**
 * @brief Saves all unique tokens (words and punctuation) to a single-column CSV file.
 * The keys from the provided map are used as the tokens.
 * @param corpus_word_counts The map containing all unique tokens as keys.
 * @param outputPath The path where the CSV file will be saved.
 */
void tokeniser::saveUniqueTokensToCSV(const std::unordered_map<std::string, int>& corpus_word_counts, const std::string& outputPath) {
    if (outputPath.empty()) {
        std::cout << "-> Output path is empty. Skipping saving unique tokens CSV." << std::endl;
        return;
    }

    std::cout << "-> Saving " << corpus_word_counts.size() << " unique tokens to: " << outputPath << std::endl;

    // Use std::ofstream for modern, safe file handling
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        // Use std::cerr for error messages
        std::cerr << "Error: Could not open file to save unique tokens: " << outputPath << std::endl;
        // It's better to throw an exception if saving is critical, or just return if it's optional.
        throw std::runtime_error("Failed to open file at: " + outputPath);
    }

    // Write the CSV header
    outFile << "token\n";

    // Iterate through the map and write each key (the token) to the file
    for (const auto& pair : corpus_word_counts) {
        const std::string& token = pair.first;
        // Handle tokens that might contain commas or quotes by enclosing them in double quotes.
        // First, escape any existing double quotes within the token itself.
        std::string escaped_token;
        for (char c : token) {
            if (c == '"') {
                escaped_token += "\"\""; // CSV standard for escaping a quote is to double it
            } else {
                escaped_token += c;
            }
        }
        outFile << "\"" << escaped_token << "\"\n";
    }

    outFile.close();
    std::cout << "-> Successfully saved unique tokens file." << std::endl;
}