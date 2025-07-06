#include "include/tokenise.hpp"
#include <iostream>
#include <algorithm>

/**
 * @brief MODIFIED: Learns a BPE vocabulary from a pre-tokenized corpus.
 * This function implements the BPE training algorithm. It distinguishes between
 * words (which are subject to BPE) and punctuation (which are treated as atomic tokens).
 * @param pre_tokens A vector of words and punctuation from the training corpus.
 * @param num_merges The number of merge operations to perform on the words.
 * @param final_vocab Output vector to store the learned vocabulary tokens.
 */
void tokeniser::groupCommonTokens(std::vector<std::string>& pre_tokens, int num_merges, std::vector<std::string>& final_vocab) {
    // 1. Initial Vocabulary & Word Preparation
    std::set<std::string> vocab;
    std::map<std::string, int> word_counts;
    std::map<std::string, std::vector<std::string>> splits;

    // Helper lambda to check if a token is a word to be split.
    auto is_word_for_bpe = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        // A simple check: does it start with a letter?
        return std::isalpha(static_cast<unsigned char>(s[0]));
    };

    // Iterate through pre_tokens, separating words for BPE
    for (const auto& token : pre_tokens) {
        if (is_word_for_bpe(token)) {
            word_counts[token]++;
        } else {
            // It's punctuation or another symbol, treat it as an atomic token.
            vocab.insert(token);
        }
    }

    // Initialize splits for each word into individual characters
    for (const auto& pair : word_counts) {
        const std::string& w = pair.first;
        std::vector<std::string> chars;
        for (char c : w) {
            std::string s(1, c);
            chars.push_back(s);
            vocab.insert(s); // Add each character to the initial vocab
        }
        // Add a special end-of-word token.
        chars.push_back("</w>");
        splits[w] = chars;
    }
    // Add the end-of-word token to the vocabulary as well.
    vocab.insert("</w>");

    // 2. Iterative Merging (This part only operates on the words)
    for (int i = 0; i < num_merges; ++i) {
        auto pair_stats = get_pair_stats(word_counts, splits);
        if (pair_stats.empty()) {
            break; 
        }

        auto best_pair = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->first;

        merge_pair(best_pair, splits);
        vocab.insert(best_pair.first + best_pair.second);

        // Optional: Keep the print statement for debugging
        std::cout << "Merge " << i + 1 << ": " << best_pair.first << " + " << best_pair.second << " -> " << best_pair.first + best_pair.second << std::endl;
    }

    final_vocab.assign(vocab.begin(), vocab.end());

    std::sort(final_vocab.begin(), final_vocab.end(), [](const std::string& a, const std::string& b){
        return a.length() > b.length();
    });

    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
}


/**
 * @brief Learns a BPE vocabulary using an incremental update algorithm.
 * This version first computes word frequencies from the flat `pre_tokens` list.
 * It then uses an inverted index to track which words are affected by a merge,
 * avoiding the massive overhead of rebuilding stats from scratch in every iteration.
 * @param pre_tokens A flat vector of all words and punctuation from the training corpus.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens.
 */
void tokeniser::groupCommonTokensParallel(std::vector<std::string>& pre_tokens, int num_merges, std::vector<std::string>& final_vocab) {
    // =========================================================================
    // 1. INITIAL SETUP
    // =========================================================================
    std::cout << "[INFO] Building initial word counts from pre-tokens..." << std::endl;
    std::set<std::string> vocab;
    std::map<std::string, int> word_counts; // This is the equivalent of corpus_word_counts
    std::map<std::string, std::vector<std::string>> splits;

    auto is_word_for_bpe = [](const std::string& s) {
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0])) && s.length() > 1;
    };

    // --- Step 1a: Build word_counts map from the flat pre_tokens list ---
    for (const auto& token : pre_tokens) {
        word_counts[token]++;
    }

    // --- Step 1b: Separate into BPE candidates and atomic tokens ---
    std::map<std::string, int> bpe_word_counts;
    for (const auto& pair : word_counts) {
        if (is_word_for_bpe(pair.first)) {
            bpe_word_counts[pair.first] = pair.second;
        } else {
            vocab.insert(pair.first);
        }
    }
    std::cout << "[INFO] Words selected for BPE processing: " << bpe_word_counts.size() << std::endl;
    if (bpe_word_counts.empty()) {
        std::cerr << "[WARNING] No words were long enough for BPE splitting." << std::endl;
        final_vocab.assign(vocab.begin(), vocab.end());
        this->tokens = final_vocab;
        this->vocSize = this->tokens.size();
        return;
    }

    // --- Step 1c: Create initial character-level splits ---
    for (const auto& pair : bpe_word_counts) {
        const std::string& word = pair.first;
        std::vector<std::string> chars;
        for (char c : word) {
            std::string s(1, c);
            chars.push_back(s);
            vocab.insert(s);
        }
        chars.push_back("</w>");
        splits[word] = std::move(chars);
    }
    vocab.insert("</w>");

    // =========================================================================
    // 2. BUILD INITIAL STATS AND INVERTED INDEX (ONCE!)
    // =========================================================================
    std::cout << "Building initial statistics and inverted index..." << std::endl;
    std::map<std::pair<std::string, std::string>, int> pair_stats;
    std::map<std::pair<std::string, std::string>, std::vector<std::string>> inverted_index;

    for (const auto& p : splits) {
        const std::string& word = p.first;
        const auto& symbols = p.second;
        if (symbols.size() < 2) continue;
        for (size_t i = 0; i < symbols.size() - 1; ++i) {
            const auto current_pair = std::make_pair(symbols[i], symbols[i + 1]);
            pair_stats[current_pair] += bpe_word_counts.at(word);
            inverted_index[current_pair].push_back(word);
        }
    }
    std::cout << "Initialization complete. Starting " << num_merges << " merges..." << std::endl;

    // =========================================================================
    // 3. HIGH-SPEED MERGE LOOP
    // =========================================================================
    for (int i = 0; i < num_merges; ++i) {
        if (pair_stats.empty()) {
            std::cout << "[INFO] No more pairs to merge. Stopping at merge " << i + 1 << "." << std::endl;
            break;
        }

        auto best_pair_it = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        const auto best_pair = best_pair_it->first;
        const int best_pair_freq = best_pair_it->second;

        const std::string new_token = best_pair.first + best_pair.second;
        vocab.insert(new_token);

        if (inverted_index.find(best_pair) == inverted_index.end()) {
            pair_stats.erase(best_pair_it);
            continue;
        }
        const auto& affected_words = inverted_index.at(best_pair);

        // This fast loop iterates ONLY over the small subset of affected words.
        for (const auto& word : affected_words) {
            auto& symbols = splits.at(word);
            int freq = bpe_word_counts.at(word);
            if (symbols.size() < 2) continue;

            std::vector<std::string> new_symbols;
            new_symbols.reserve(symbols.size());

            size_t k = 0;
            while (k < symbols.size()) {
                if (k < symbols.size() - 1 && symbols[k] == best_pair.first && symbols[k + 1] == best_pair.second) {
                    // Update stats for surrounding pairs
                    if (k > 0) {
                        auto old_left_pair = std::make_pair(symbols[k - 1], best_pair.first);
                        pair_stats[old_left_pair] -= freq;
                        if (pair_stats[old_left_pair] <= 0) pair_stats.erase(old_left_pair);
                        
                        auto new_left_pair = std::make_pair(symbols[k - 1], new_token);
                        pair_stats[new_left_pair] += freq;
                        inverted_index[new_left_pair].push_back(word);
                    }
                    if (k < symbols.size() - 2) {
                        auto old_right_pair = std::make_pair(best_pair.second, symbols[k + 2]);
                        pair_stats[old_right_pair] -= freq;
                        if (pair_stats[old_right_pair] <= 0) pair_stats.erase(old_right_pair);

                        auto new_right_pair = std::make_pair(new_token, symbols[k + 2]);
                        pair_stats[new_right_pair] += freq;
                        inverted_index[new_right_pair].push_back(word);
                    }
                    new_symbols.push_back(new_token);
                    k += 2;
                } else {
                    new_symbols.push_back(symbols[k]);
                    k += 1;
                }
            }
            symbols = std::move(new_symbols);
        }

        inverted_index.erase(best_pair);
        pair_stats.erase(best_pair);

        if ((i + 1) % 1000 == 0 || i == num_merges - 1) {
            std::cout << "Merge " << i + 1 << "/" << num_merges << ": Merged '" << best_pair.first 
                      << "' and '" << best_pair.second << "' (Frequency: " << best_pair_freq << ")" << std::endl;
        }
    }

    // =========================================================================
    // 4. FINALIZE VOCABULARY
    // =========================================================================
    final_vocab.assign(vocab.begin(), vocab.end());
    std::sort(final_vocab.begin(), final_vocab.end(), [](const auto& a, const auto& b){ return a.length() > b.length(); });

    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
    std::cout << "\nBPE training complete. Final vocabulary size: " << this->vocSize << std::endl;
}


/**
 * @brief (INVERTED INDEX) Learns a BPE vocabulary.
 * This version uses an inverted index to track which words are affected by a merge,
 * providing a massive speedup. It includes corrected initialization logic and robust
 * debugging logs to verify each stage of the process.
 * @param corpus_word_counts A map of unique words and their frequencies.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens.
 */
void tokeniser::groupCommonTokensParallel(const std::map<std::string, int>& corpus_word_counts, int num_merges,
    std::vector<std::string>& final_vocab)
{
    // =========================================================================
    // 1. INITIAL SETUP
    // =========================================================================
    std::set<std::string> vocab;
    std::map<std::string, int> bpe_word_counts;
    std::map<std::string, std::vector<std::string>> splits;

    // A robust check for what constitutes a "word" to be split by BPE.
    auto is_word_for_bpe = [](const std::string& s) {
        // Must not be empty, must start with a letter, and must be longer than 1 character
        // to be eligible for splitting. Single-letter words are atomic.
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0])) && s.length() > 1;
    };

    // --- Step 1a: Separate raw tokens into BPE candidates and atomic tokens ---
    std::cout << "[INFO] Total unique raw tokens received: " << corpus_word_counts.size() << std::endl;
    for (const auto& pair : corpus_word_counts) {
        if (is_word_for_bpe(pair.first)) {
            bpe_word_counts[pair.first] = pair.second;
        } else {
            // This includes punctuation, symbols, and single-letter words.
            vocab.insert(pair.first);
        }
    }
    std::cout << "[INFO] Words selected for BPE processing: " << bpe_word_counts.size() << std::endl;
    std::cout << "[INFO] Initial atomic tokens (punctuation, etc.): " << vocab.size() << std::endl;
    std::cout << "[INFO] Number of BPE merges to perform: " << num_merges << std::endl;

    // --- Sanity Check: Exit gracefully if no words are suitable for BPE ---
    if (bpe_word_counts.empty()) {
        std::cerr << "[WARNING] No words were suitable for BPE splitting. Vocabulary will only contain initial atomic tokens." << std::endl;
        final_vocab.assign(vocab.begin(), vocab.end());
        this->tokens = final_vocab;
        this->vocSize = this->tokens.size();
        return;
    }

    // --- Step 1b: Create initial character-level splits and populate base vocabulary ---
    for (const auto& pair : bpe_word_counts) {
        const std::string& word = pair.first;
        std::vector<std::string> chars;
        for (char c : word) {
            std::string s(1, c);
            chars.push_back(s);
            vocab.insert(s); // Add every single character to the initial vocab
        }
        chars.push_back("</w>"); // Add the essential end-of-word token
        splits[word] = std::move(chars);
    }
    vocab.insert("</w>"); // Ensure end-of-word token is in the vocab

    // =========================================================================
    // 2. BUILD INITIAL STATS AND INVERTED INDEX (ONCE!)
    // =========================================================================
    std::cout << "[INFO] Building initial pair statistics and inverted index..." << std::endl;
    std::map<std::pair<std::string, std::string>, int> pair_stats;
    std::map<std::pair<std::string, std::string>, std::vector<std::string>> inverted_index;

    for (const auto& p : splits) {
        const std::string& word = p.first;
        const auto& symbols = p.second;
        if (symbols.size() < 2) continue;
        for (size_t i = 0; i < symbols.size() - 1; ++i) {
            const auto current_pair = std::make_pair(symbols[i], symbols[i + 1]);
            pair_stats[current_pair] += bpe_word_counts.at(word);
            inverted_index[current_pair].push_back(word);
        }
    }

    std::cout << "[INFO] Initialization complete. Initial pair_stats size: " << pair_stats.size() << ". Starting merges..." << std::endl;

    // =========================================================================
    // 3. HIGH-SPEED MERGE LOOP
    // =========================================================================
    for (int i = 0; i < num_merges; ++i) {
        if (pair_stats.empty()) {
            std::cout << "[INFO] No more pairs to merge. Stopping at merge " << i + 1 << "." << std::endl;
            break;
        }

        auto best_pair_it = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        const auto best_pair = best_pair_it->first;
        const int best_pair_freq = best_pair_it->second;
        const std::string new_token = best_pair.first + best_pair.second;
        
        vocab.insert(new_token);

        // Check if the best pair exists in the index (it always should, but this is a safe guard)
        if (inverted_index.find(best_pair) == inverted_index.end()) {
            pair_stats.erase(best_pair_it); // Clean up inconsistent stat and continue
            continue;
        }
        
        // This is the key optimization: we only iterate over the words that actually contain the best_pair
        const auto& affected_words = inverted_index.at(best_pair);

        for (const auto& word : affected_words) {
            auto& symbols = splits.at(word);
            int freq = bpe_word_counts.at(word);
            if (symbols.size() < 2) continue;

            std::vector<std::string> new_symbols;
            new_symbols.reserve(symbols.size());

            size_t k = 0;
            while (k < symbols.size()) {
                // Find the merge location
                if (k < symbols.size() - 1 && symbols[k] == best_pair.first && symbols[k + 1] == best_pair.second) {
                    // This is the merge point. We must update stats for pairs that are being destroyed and created.
                    
                    // 1. Decrement stats for the pair to the LEFT of the merge
                    if (k > 0) {
                        auto old_left_pair = std::make_pair(symbols[k - 1], best_pair.first);
                        pair_stats[old_left_pair] -= freq;
                        if (pair_stats[old_left_pair] <= 0) pair_stats.erase(old_left_pair);
                        
                        // Increment stats for the NEW pair on the left
                        auto new_left_pair = std::make_pair(symbols[k - 1], new_token);
                        pair_stats[new_left_pair] += freq;
                        inverted_index[new_left_pair].push_back(word);
                    }
                    
                    // 2. Decrement stats for the pair to the RIGHT of the merge
                    if (k < symbols.size() - 2) {
                        auto old_right_pair = std::make_pair(best_pair.second, symbols[k + 2]);
                        pair_stats[old_right_pair] -= freq;
                        if (pair_stats[old_right_pair] <= 0) pair_stats.erase(old_right_pair);

                        // Increment stats for the NEW pair on the right
                        auto new_right_pair = std::make_pair(new_token, symbols[k + 2]);
                        pair_stats[new_right_pair] += freq;
                        inverted_index[new_right_pair].push_back(word);
                    }

                    // Perform the merge in the new symbol list
                    new_symbols.push_back(new_token);
                    k += 2;
                } else {
                    new_symbols.push_back(symbols[k]);
                    k += 1;
                }
            }
            symbols = std::move(new_symbols);
        }

        // Clean up the master lists for the pair we just merged
        inverted_index.erase(best_pair);
        pair_stats.erase(best_pair);

        // Logging progress
        if ((i + 1) % 1000 == 0 || i == num_merges - 1) {
            std::cout << " -> Merge " << i + 1 << "/" << num_merges << ": Merged '" << best_pair.first
                      << "' + '" << best_pair.second << "' -> '" << new_token << "' (Freq: " << best_pair_freq
                      << ", Vocab size: " << vocab.size() << ")" << std::endl;
        }
    }

    // =========================================================================
    // 4. FINALIZE VOCABULARY
    // =========================================================================
    final_vocab.assign(vocab.begin(), vocab.end());
    // Sort by length (descending) to ensure the tokenizer picks the longest possible match
    std::sort(final_vocab.begin(), final_vocab.end(), [](const auto& a, const auto& b){
        return a.length() > b.length();
    });

    // Update the tokenizer's internal state
    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
    std::cout << "\n[SUCCESS] BPE training complete. Final vocabulary size: " << this->vocSize << std::endl;
}
