
#include "include/tokenise.hpp"
#include <future>
#include <fstream>
#include <sstream>
#include <regex>
#include <map>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <thread>
#include <mutex>


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
 * @brief Calculates pair statistics in parallel.
 * Splits the workload (words) across multiple threads. Each thread computes stats
 * for its chunk, and the results are merged at the end.
 */
std::map<std::pair<std::string, std::string>, int> parallel_get_pair_stats(
    const std::map<std::string, int>& word_counts,
    const std::map<std::string, std::vector<std::string>>& splits,
    int num_threads)
{
    if (num_threads <= 1) {
        return get_pair_stats(word_counts, splits); // Fallback to serial version
    }

    std::vector<std::string> words;
    words.reserve(word_counts.size());
    for(const auto& pair : word_counts) {
        words.push_back(pair.first);
    }

    std::vector<std::future<std::map<std::pair<std::string, std::string>, int>>> futures;
    int chunk_size = (words.size() + num_threads - 1) / num_threads; // Ceiling division

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, (int)words.size());
        if (start >= end) continue;

        futures.push_back(std::async(std::launch::async, [start, end, &words, &word_counts, &splits] {
            std::map<std::pair<std::string, std::string>, int> local_pair_freqs;
            for (int j = start; j < end; ++j) {
                const std::string& word = words[j];
                const int count = word_counts.at(word);
                const std::vector<std::string>& symbols = splits.at(word);
                for (size_t k = 0; k < symbols.size() - 1; ++k) {
                    local_pair_freqs[{symbols[k], symbols[k + 1]}] += count;
                }
            }
            return local_pair_freqs;
        }));
    }

    // Merge results
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
 * @brief Merges the best pair across all word splits in parallel.
 * Splits the workload (words) across multiple threads. Since each thread modifies
 * a different word's split in the map, this operation is safe without a mutex.
 */
void parallel_merge_pair(
    const std::pair<std::string, std::string>& best_pair,
    std::map<std::string, std::vector<std::string>>& splits,
    int num_threads)
{
    if (num_threads <= 1) {
        merge_pair(best_pair, splits); // Fallback to serial version
        return;
    }

    std::vector<std::string> words_to_process;
    words_to_process.reserve(splits.size());
    for(const auto& pair : splits) {
        words_to_process.push_back(pair.first);
    }

    std::vector<std::thread> threads;
    int chunk_size = (words_to_process.size() + num_threads - 1) / num_threads;

    std::string new_token = best_pair.first + best_pair.second;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, (int)words_to_process.size());
        if (start >= end) continue;

        threads.emplace_back([start, end, &words_to_process, &splits, &best_pair, &new_token] {
            for (int j = start; j < end; ++j) {
                const std::string& word = words_to_process[j];
                std::vector<std::string>& symbols = splits.at(word);
                if (symbols.size() < 2) continue;

                std::vector<std::string> new_symbols;
                new_symbols.reserve(symbols.size());
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
                symbols = new_symbols;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// --- Class Method Implementations ---

/**
 * @brief Extracts all words from a text file.
 * Reads a file, finds all sequences of alphabetic characters, converts them
 * to lowercase, and populates a vector with them.
 * @param path2txt The path to the input text file.
 * @param words Output vector to store the extracted words.
 */
void tokeniser::splitWordsFromTxt(const std::string& path2txt, std::vector<std::string>& pre_tokens) {
    pre_tokens.clear();
    std::ifstream file(path2txt);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + path2txt);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    // MODIFICATION: Use the more sophisticated regex to capture words OR punctuation.
    // This regex captures sequences of letters OR single non-letter, non-space characters.
    std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");
    auto tokens_begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto tokens_end = std::sregex_iterator();

    for (std::sregex_iterator i = tokens_begin; i != tokens_end; ++i) {
        std::string token = (*i).str();
        // MODIFICATION: Only convert to lowercase if it's a word.
        if (!token.empty() && std::isalpha(static_cast<unsigned char>(token[0]))) {
            std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        }
        pre_tokens.push_back(token);
    }
}


/**
 * @brief Extracts all words from a text file in parallel.
 * Reads a large text file, divides its content among multiple threads, and
 * tokenizes each chunk concurrently using regex.
 * @param path2txt The path to the input text file.
 * @param pre_tokens Output vector to store the extracted words and symbols.
 */
void tokeniser::splitWordsFromTxtParallel(const std::string& path2txt, std::vector<std::string>& pre_tokens) {
    pre_tokens.clear();
    std::ifstream file(path2txt);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + path2txt);
    }

    // 1. Read the entire file into memory (fastest for I/O)
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    if (text.empty()) {
        return;
    }

    int n_threads = this->num_threads;
    if (n_threads <= 1 || text.length() < 10000) { // Fallback for single thread or small files
        // Use original serial implementation for small files where threading overhead isn't worth it
        std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");
        auto tokens_begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto tokens_end = std::sregex_iterator();
        for (std::sregex_iterator i = tokens_begin; i != tokens_end; ++i) {
            std::string token = (*i).str();
            if (!token.empty() && std::isalpha(static_cast<unsigned char>(token[0]))) {
                std::transform(token.begin(), token.end(), token.begin(), ::tolower);
            }
            pre_tokens.push_back(token);
        }
        return;
    }

    // 2. Determine chunk boundaries, ensuring we don't split words
    std::vector<size_t> chunk_boundaries;
    chunk_boundaries.push_back(0);
    size_t chunk_size = text.length() / n_threads;

    for (int i = 1; i < n_threads; ++i) {
        size_t boundary = i * chunk_size;
        // Scan forward from the ideal boundary to find the next whitespace to avoid splitting a token
        while (boundary < text.length() && !std::isspace(static_cast<unsigned char>(text[boundary]))) {
            boundary++;
        }
        chunk_boundaries.push_back(boundary);
    }
    chunk_boundaries.push_back(text.length());


    // 3. Launch threads to process chunks in parallel
    std::vector<std::future<std::vector<std::string>>> futures;
    std::string_view text_view(text); // Use string_view to avoid copying the large string

    for (size_t i = 0; i < n_threads; ++i) {
        size_t start = chunk_boundaries[i];
        size_t end = chunk_boundaries[i+1];
        
        // Skip empty chunks
        if (start >= end) continue;

        // Find the first non-whitespace character for the start of the chunk
        while (start < end && std::isspace(static_cast<unsigned char>(text_view[start]))) {
            start++;
        }

        futures.push_back(std::async(std::launch::async, [text_view, start, end]() {
            std::vector<std::string> local_tokens;
            std::string_view sub_view = text_view.substr(start, end - start);

            // The regex to find words or single punctuation characters
            std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");

            // --- FIX IS HERE ---
            // Get raw pointers from the string_view for the regex_iterator
            const char* text_begin = sub_view.data();
            const char* text_end = text_begin + sub_view.size();

            // Now use cregex_iterator with the pointers
            auto tokens_begin = std::cregex_iterator(text_begin, text_end, re);
            auto tokens_end = std::cregex_iterator();

            for (auto it = tokens_begin; it != tokens_end; ++it) {
                std::string token = it->str();
                // Process the token (lowercase if it's a word)
                if (!token.empty() && std::isalpha(static_cast<unsigned char>(token[0]))) {
                    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
                }
                local_tokens.push_back(std::move(token)); // Use move for efficiency
            }
            return local_tokens;
        }));
    }

    // 4. Aggregate results from all threads
    for (auto& f : futures) {
        auto local_tokens = f.get();
        // Efficiently move the elements from the local vector to the final one
        pre_tokens.insert(pre_tokens.end(), 
                          std::make_move_iterator(local_tokens.begin()), 
                          std::make_move_iterator(local_tokens.end()));
    }
}


/**
 * @brief Splits a single word into a sequence of subword tokens using the learned vocabulary.
 * This function implements the tokenization of a single word by greedily matching the
 * longest possible tokens from the vocabulary.
 * @param word The word to be tokenized.
 * @param subwords Output vector to store the resulting subword tokens.
 */
void tokeniser::splitWord(const std::string& word, std::vector<std::string>& subwords) const {
    subwords.clear();
    if (word.empty()) return;

    // Add end-of-word token to handle word boundaries correctly
    std::string current_word = word + "</w>";
    
    while (!current_word.empty()) {
        bool found_match = false;
        // Greedily find the longest token in our vocabulary that is a prefix of the current word.
        // This requires the vocabulary `this->tokens` to be sorted by length, descending.
        for (const auto& token : this->tokens) {
            if (current_word.rfind(token, 0) == 0) { // check if string starts with token
                subwords.push_back(token);
                current_word = current_word.substr(token.length());
                found_match = true;
                break;
            }
        }
        if (!found_match) {
            // Fallback for unknown characters. This should not happen if the initial
            // vocabulary includes all single characters from the training corpus.
            subwords.push_back(current_word.substr(0, 1));
            current_word = current_word.substr(1);
        }
    }
}

/**
 * @brief Tokenizes a full sentence into a sequence of subword tokens.
 * This function splits a sentence into words and punctuation, then tokenizes
 * each word using the `splitWord` method.
 * @param sentence The input sentence string.
 * @param all_subwords Output vector to store the final sequence of tokens.
 */
void tokeniser::splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords) const {
    all_subwords.clear();

    // Regex to split sentence by words and punctuation.
    // It captures sequences of letters OR single non-letter, non-space characters.
    std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");
    auto words_begin = std::sregex_iterator(sentence.begin(), sentence.end(), re);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::string token_str = (*i).str();
        
        // If it's a word (starts with a letter), split it using our learned vocabulary
        if (std::isalpha(token_str[0])) {
            std::string lower_token_str = token_str;
            std::transform(lower_token_str.begin(), lower_token_str.end(), lower_token_str.begin(), ::tolower);
            
            std::vector<std::string> word_subwords;
            splitWord(lower_token_str, word_subwords);
            all_subwords.insert(all_subwords.end(), word_subwords.begin(), word_subwords.end());
        } 
        else {
            // It's punctuation or another symbol, keep it as a single token
            all_subwords.push_back(token_str);
        }
    }
}

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

// --- MODIFIED `groupCommonTokens` to use the parallel functions ---

void tokeniser::groupCommonTokensParallel(std::vector<std::string>& pre_tokens, int num_merges, std::vector<std::string>& final_vocab) {
    // 1. Initial Vocabulary & Word Preparation (this part is serial and fast)
    // ... (This section remains unchanged) ...
    std::set<std::string> vocab;
    std::map<std::string, int> word_counts;
    std::map<std::string, std::vector<std::string>> splits;

    auto is_word_for_bpe = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        return std::isalpha(static_cast<unsigned char>(s[0]));
    };

    for (const auto& token : pre_tokens) {
        if (is_word_for_bpe(token)) {
            word_counts[token]++;
        } else {
            vocab.insert(token);
        }
    }
    
    for (const auto& pair : word_counts) {
        const std::string& w = pair.first;
        std::vector<std::string> chars;
        for (char c : w) {
            std::string s(1, c);
            chars.push_back(s);
            vocab.insert(s);
        }
        chars.push_back("</w>");
        splits[w] = chars;
    }
    vocab.insert("</w>");


    // 2. Iterative Merging (MODIFIED TO BE PARALLEL)
    std::cout << "Starting BPE training with " << this->num_threads << " threads." << std::endl;
    for (int i = 0; i < num_merges; ++i) {
        // USE PARALLEL VERSION
        auto pair_stats = parallel_get_pair_stats(word_counts, splits, this->num_threads);

        if (pair_stats.empty()) {
            break;
        }

        auto best_pair = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            })->first;

        // USE PARALLEL VERSION
        parallel_merge_pair(best_pair, splits, this->num_threads);
        
        vocab.insert(best_pair.first + best_pair.second);

        std::cout << "Merge " << i + 1 << "/" << num_merges << ": " << best_pair.first << " + " << best_pair.second << " -> " << best_pair.first + best_pair.second << std::endl;
    }

    // ... (rest of the function remains the same) ...
    final_vocab.assign(vocab.begin(), vocab.end());

    std::sort(final_vocab.begin(), final_vocab.end(), [](const std::string& a, const std::string& b){
        return a.length() > b.length();
    });

    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
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


/**
 * @brief Reads a file (.txt or .csv) line by line and tokenizes its entire content.
 * This function is designed for efficiency with large files. It reads one line
 * at a time, tokenizes it using the pre-trained BPE vocabulary, and appends
 * the resulting subwords to a single output vector. For CSV files, it treats
 * each row as a single sentence to be tokenized.
 * @param filePath The path to the input file.
 * @return A flat vector of strings containing all the BPE subwords from the file.
 * @throws std::runtime_error if the file cannot be opened.
 */
std::vector<std::string> tokeniser::tokeniseFile(const std::string& filePath) const {
    // 1. Open the file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + filePath);
    }

    std::vector<std::string> all_file_subwords;
    std::string line;

    // 2. Read file line by line to handle large files efficiently
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // Optionally skip empty lines
        }

        // 3. Tokenize the current line using the existing splitSentence logic
        std::vector<std::string> line_subwords;
        this->splitSentence(line, line_subwords);

        // 4. Append the results to the master vector
        // Using insert is generally more efficient for appending a whole vector
        all_file_subwords.insert(all_file_subwords.end(), line_subwords.begin(), line_subwords.end());
    }

    file.close();
    return all_file_subwords;
}