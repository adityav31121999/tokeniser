#include "include/tokenise.hpp" // Your header file
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <regex>
#include <algorithm>
#include <future>
#include <string_view>
#include <set>
#include <cctype>
#include <chrono>
#include <filesystem>

// =================================================================================
// SECTION 1: PARALLEL BPE HELPER FUNCTIONS
// These are the core parallel routines for the BPE training loop.
// =================================================================================

/**
 * @brief (Parallel) Calculates the frequency of all adjacent pairs of symbols.
 * Splits the workload (unique words) across multiple threads for concurrent processing.
 *
 * @param word_counts Map from each word to its frequency.
 * @param splits Map from each word to its current token representation.
 * @param num_threads Number of threads to use.
 * @return A map of pair frequencies, aggregated from all threads.
 */
std::map<std::pair<std::string, std::string>, int> parallel_get_pair_stats(
    const std::map<std::string, int>& word_counts,
    const std::map<std::string, std::vector<std::string>>& splits,
    int num_threads)
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
 * @brief (Parallel) Merges a specified pair of tokens into a new single token across all word splits.
 * This is highly parallelizable as each word's split is independent.
 *
 * @param best_pair The pair of tokens to be merged.
 * @param splits The map of word splits to be updated in place.
 * @param num_threads Number of threads to use.
 */
void parallel_merge_pair(
    const std::pair<std::string, std::string>& best_pair,
    std::map<std::string, std::vector<std::string>>& splits,
    int num_threads)
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


// =================================================================================
// SECTION 2: TOKENISER CLASS METHOD IMPLEMENTATIONS (OPTIMIZED)
// =================================================================================

/**
 * @brief (Parallel) Extracts tokens from a large text file using multiple threads.
 * Reads the entire file, splits it into chunks, and processes them in parallel.
 * @param path2txt The path to the input text file.
 * @param pre_tokens Output vector to store the extracted words and symbols.
 */
void tokeniser::splitWordsFromTxtParallel(const std::string& path2txt, std::vector<std::string>& pre_tokens) {
    pre_tokens.clear();
    std::ifstream file(path2txt);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + path2txt);
    }

    // 1. Read the entire file into memory for fastest access.
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    file.close();

    if (text.empty()) {
        return;
    }

    // Fallback to serial for small files where threading overhead is not worth it.
    if (this->num_threads <= 1 || text.length() < 100000) { // Threshold of 100KB
        splitWordsFromTxt(path2txt, pre_tokens);
        return;
    }

    // 2. Determine chunk boundaries, ensuring we don't split words.
    std::vector<size_t> chunk_starts;
    chunk_starts.push_back(0);
    size_t approx_chunk_size = text.length() / this->num_threads;

    for (int i = 1; i < this->num_threads; ++i) {
        size_t boundary = i * approx_chunk_size;
        // Scan forward to find the next whitespace to avoid splitting a token.
        while (boundary < text.length() && !std::isspace(static_cast<unsigned char>(text[boundary]))) {
            boundary++;
        }
        if (boundary < text.length()) {
            chunk_starts.push_back(boundary);
        }
    }
    
    // 3. Launch threads to process chunks in parallel.
    std::vector<std::future<std::vector<std::string>>> futures;
    const std::string_view text_view(text);

    for (size_t i = 0; i < chunk_starts.size(); ++i) {
        size_t start = chunk_starts[i];
        size_t end = (i + 1 < chunk_starts.size()) ? chunk_starts[i+1] : text.length();
        
        // Skip empty chunks
        if (start >= end) continue;

        // Adjust start to skip leading whitespace in the chunk
        while (start < end && std::isspace(static_cast<unsigned char>(text_view[start]))) {
            start++;
        }
        if (start >= end) continue;

        futures.push_back(std::async(std::launch::async, [text_view, start, end]() {
            std::vector<std::string> local_tokens;
            std::string_view sub_view = text_view.substr(start, end - start);
            
            // Regex to find words or single punctuation characters
            const std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");

            // *** CRITICAL FIX ***
            // Use cregex_iterator with raw pointers, as string_view iterators are not compatible.
            auto tokens_begin = std::cregex_iterator(sub_view.data(), sub_view.data() + sub_view.size(), re);
            auto tokens_end = std::cregex_iterator();

            for (auto it = tokens_begin; it != tokens_end; ++it) {
                std::string token = it->str();
                if (!token.empty() && std::isalpha(static_cast<unsigned char>(token[0]))) {
                    std::transform(token.begin(), token.end(), token.begin(),
                                   [](unsigned char c){ return std::tolower(c); });
                }
                local_tokens.push_back(std::move(token));
            }
            return local_tokens;
        }));
    }

    // 4. Aggregate results from all threads.
    for (auto& f : futures) {
        auto local_tokens = f.get();
        pre_tokens.insert(pre_tokens.end(), 
                          std::make_move_iterator(local_tokens.begin()), 
                          std::make_move_iterator(local_tokens.end()));
    }
}


/**
 * @brief (Parallel) Learns a BPE vocabulary from a pre-tokenized corpus using multiple threads.
 *
 * @param pre_tokens A vector of words and punctuation from the training corpus.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens.
 */
void tokeniser::groupCommonTokensParallel(std::vector<std::string>& pre_tokens, int num_merges, std::vector<std::string>& final_vocab) {
    // 1. Initial Vocabulary & Word Preparation (this part is fast and remains serial)
    std::set<std::string> vocab;
    std::map<std::string, int> word_counts;
    std::map<std::string, std::vector<std::string>> splits;

    auto is_word_for_bpe = [](const std::string& s) -> bool {
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0]));
    };

    for (const auto& token : pre_tokens) {
        if (is_word_for_bpe(token)) {
            word_counts[token]++;
        } else {
            vocab.insert(token); // Punctuation/symbols are atomic tokens
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
        chars.push_back("</w>"); // Special end-of-word token
        splits[w] = std::move(chars);
    }
    vocab.insert("</w>");

    // 2. Iterative Merging (THE COMPUTATIONALLY EXPENSIVE PART - NOW PARALLEL)
    std::cout << "Starting BPE training with " << this->num_threads << " threads for " << num_merges << " merges." << std::endl;
    for (int i = 0; i < num_merges; ++i) {
        // --- Step 2a: Calculate pair stats in parallel ---
        auto pair_stats = parallel_get_pair_stats(word_counts, splits, this->num_threads);

        if (pair_stats.empty()) {
            std::cout << "No more pairs to merge. Stopping at merge " << i << "." << std::endl;
            break; 
        }

        // --- Step 2b: Find the best pair (this is fast) ---
        auto best_pair_it = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
        const auto& best_pair = best_pair_it->first;

        // --- Step 2c: Merge the best pair in parallel ---
        parallel_merge_pair(best_pair, splits, this->num_threads);
        
        vocab.insert(best_pair.first + best_pair.second);

        if ((i + 1) % 10 == 0 || i == num_merges - 1) { // Print progress periodically
            std::cout << "Merge " << i + 1 << "/" << num_merges << ": Merged '" << best_pair.first 
                      << "' and '" << best_pair.second << "' (Frequency: " << best_pair_it->second << ")" << std::endl;
        }
    }

    // 3. Finalize vocabulary (fast, remains serial)
    final_vocab.assign(vocab.begin(), vocab.end());

    // Sort by length (descending) for greedy matching in `splitWord`
    std::sort(final_vocab.begin(), final_vocab.end(), [](const std::string& a, const std::string& b){
        return a.length() > b.length();
    });

    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
    std::cout << "BPE training complete. Final vocabulary size: " << this->vocSize << std::endl;
}

// =================================================================================
// SECTION 5: NEW HIGH-PERFORMANCE WORKFLOW
// =================================================================================

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

// =========================================================================

#include <iomanip>    // For std::setw, std::left

// Create a global mutex and progress map for debugging.
std::mutex progress_mutex;
std::map<std::string, int> progress_map;


/**
 * @brief (EVENT-DRIVEN LOGGING) Builds word counts with progress reporting tied to file completion.
 * This version uses the producer-consumer model and an event-driven logging
 * system. The main thread waits on a condition variable and prints a progress
 * update only when the producer thread signals that it has finished processing a file.
 * @param file_paths A vector of paths to the text files.
 * @param corpus_word_counts Output map to be filled with unique tokens and their total counts.
 */
void tokeniser::buildCorpusWordCountsParallel(const std::vector<std::string>& file_paths, std::map<std::string, int>& corpus_word_counts) {
    corpus_word_counts.clear();

    const size_t CHUNK_SIZE = 10000;
    ThreadSafeQueue<std::vector<std::string>> work_queue;
    int num_consumers = this->num_threads > 1 ? this->num_threads - 1 : 1;
    ProgressData progress;

    // 1. PRE-COMPUTATION
    for (const auto& path : file_paths) {
        if (std::filesystem::exists(path)) {
            progress.total_bytes += std::filesystem::file_size(path);
        }
    }

    // 2. DEFINE THE CONSUMER'S JOB (unchanged)
    auto consumer_task = [&work_queue]() -> std::unordered_map<std::string, int> {
        // ... (This lambda is identical to the previous version, no changes needed) ...
        std::unordered_map<std::string, int> local_counts;
        std::vector<std::string> chunk;
        while (work_queue.wait_and_pop(chunk)) {
            for (const auto& line : chunk) {
                for (size_t i = 0; i < line.length(); ) {
                    unsigned char current_char = line[i];
                    if (std::isalpha(current_char)) {
                        size_t start = i;
                        while (i < line.length() && std::isalpha(static_cast<unsigned char>(line[i]))) i++;
                        std::string word = line.substr(start, i - start);
                        std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c){ return std::tolower(c); });
                        local_counts[word]++;
                    } else if (!std::isspace(current_char)) {
                        local_counts[line.substr(i, 1)]++;
                        i++;
                    } else { i++; }
                }
            }
        }
        return local_counts;
    };

    // 3. LAUNCH THREADS
    std::cout << "-> Launching 1 Producer and " << num_consumers << " Consumer threads..." << std::endl;

    std::vector<std::future<std::unordered_map<std::string, int>>> consumer_futures;
    for (int i = 0; i < num_consumers; ++i) {
        consumer_futures.push_back(std::async(std::launch::async, consumer_task));
    }

    // Producer thread now uses the condition variable to signal progress.
    std::future<void> producer_future = std::async(std::launch::async, [&]() {
        std::vector<std::string> chunk_buffer;
        chunk_buffer.reserve(CHUNK_SIZE);

        for (const auto& path : file_paths) {
            long long file_bytes_read = 0;
            {
                std::lock_guard<std::mutex> lock(progress.mtx);
                progress.current_file_in_progress = std::filesystem::path(path).filename().string();
            }
            
            std::ifstream file(path);
            std::string line;
            while (std::getline(file, line)) {
                file_bytes_read += line.length() + 1; // +1 for newline
                chunk_buffer.push_back(std::move(line));
                if (chunk_buffer.size() >= CHUNK_SIZE) {
                    work_queue.push(std::move(chunk_buffer));
                    chunk_buffer.clear();
                    chunk_buffer.reserve(CHUNK_SIZE);
                }
            }

            // --- CRITICAL CHANGE: Signal after each file ---
            {
                std::unique_lock<std::mutex> lock(progress.mtx);
                progress.bytes_read += file_bytes_read;
                progress.files_completed_count++;
                // Notify the main thread that a file is done.
                progress.cv.notify_one();
            }
        }

        if (!chunk_buffer.empty()) {
            work_queue.push(std::move(chunk_buffer));
        }
        work_queue.close();
    });

    // 4. MAIN THREAD EVENT-DRIVEN LOGGING LOOP
    {
        std::unique_lock<std::mutex> lock(progress.mtx);
        size_t total_files = file_paths.size();

        // Loop until all files are marked as completed by the producer.
        while (progress.files_completed_count < total_files) {
            // Wait for a signal from the producer.
            // The cv.wait will unlock the mutex and wait. When woken up, it re-acquires the lock.
            progress.cv.wait(lock);

            // When we wake up, we know progress has been made. Print the report.
            double percentage = 0.0;
            if (progress.total_bytes > 0) {
                percentage = static_cast<double>(progress.bytes_read) / progress.total_bytes * 100.0;
            }

            std::cout << "  -> Progress: [" << std::fixed << std::setprecision(2) << percentage << "%] "
                      << "| Completed " << progress.files_completed_count << "/" << total_files << " files. "
                      << "(Finished '" << progress.current_file_in_progress << "')" << std::endl;
        }
    }
    std::cout << "-> Producer has finished reading all files. Waiting for consumers...\n";


    // 5. AGGREGATE FINAL RESULTS (same as before)
    std::unordered_map<std::string, int> final_counts;
    for (auto& f : consumer_futures) {
        auto local_map = f.get();
        for (const auto& pair : local_map) {
            final_counts[pair.first] += pair.second;
        }
    }

    producer_future.get();

    corpus_word_counts.clear();
    corpus_word_counts.insert(final_counts.begin(), final_counts.end());
}


/**
 * @brief (OPTIMIZED) Learns a BPE vocabulary using an incremental update algorithm.
 *
 * This version calculates pair statistics only once. In each merge step, it finds
 * the best pair and then incrementally updates the statistics map and the word
s * splits in parallel. This avoids the massive overhead of rebuilding stats from
 * scratch in every iteration, making it orders of magnitude faster.
 *
 * @param corpus_word_counts A map of unique words and their frequencies.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens.
 */
void tokeniser::groupCommonTokensParallel(const std::map<std::string, int>& corpus_word_counts, int num_merges, std::vector<std::string>& final_vocab) {
    // 1. INITIAL SETUP (largely the same)
    std::set<std::string> vocab;
    std::map<std::string, int> bpe_word_counts;
    std::map<std::string, std::vector<std::string>> splits;

    auto is_word_for_bpe = [](const std::string& s) -> bool {
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0]));
    };

    for (const auto& pair : corpus_word_counts) {
        if (is_word_for_bpe(pair.first)) {
            bpe_word_counts[pair.first] = pair.second;
        } else {
            vocab.insert(pair.first);
        }
    }

    for (const auto& pair : bpe_word_counts) {
        const std::string& w = pair.first;
        std::vector<std::string> chars;
        for (char c : w) {
            std::string s(1, c);
            chars.push_back(s);
            vocab.insert(s);
        }
        chars.push_back("</w>");
        splits[w] = std::move(chars);
    }
    vocab.insert("</w>");

    // 2. CALCULATE INITIAL STATS (ONCE!)
    std::cout << "Calculating initial pair statistics..." << std::endl;
    std::map<std::pair<std::string, std::string>, int> pair_stats = 
        parallel_get_pair_stats(bpe_word_counts, splits, this->num_threads);
    std::cout << "Initial statistics calculated. Starting merges." << std::endl;


    // 3. OPTIMIZED MERGE LOOP
    for (int i = 0; i < num_merges; ++i) {
        if (pair_stats.empty()) {
            std::cout << "No more pairs to merge. Stopping at merge " << i + 1 << "." << std::endl;
            break;
        }

        // --- Step 3a: Find best pair (extremely fast) ---
        auto best_pair_it = std::max_element(pair_stats.begin(), pair_stats.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        const auto best_pair = best_pair_it->first;
        const int best_pair_freq = best_pair_it->second;

        // --- Step 3b: Update vocab and remove the just-merged pair ---
        const std::string new_token = best_pair.first + best_pair.second;
        vocab.insert(new_token);
        pair_stats.erase(best_pair_it);

        // --- Step 3c: Update splits and stats incrementally and in parallel ---
        // This is the new core logic. We iterate over all words and update ONLY where the best_pair occurs.
        std::vector<std::string> words_to_process;
        words_to_process.reserve(bpe_word_counts.size());
        for(const auto& p : bpe_word_counts) words_to_process.push_back(p.first);

        std::mutex stats_mutex; // Mutex to protect the shared pair_stats map
        std::vector<std::thread> threads;
        int chunk_size = (words_to_process.size() + this->num_threads - 1) / this->num_threads;

        for (int t = 0; t < this->num_threads; ++t) {
            int start_idx = t * chunk_size;
            int end_idx = std::min(start_idx + chunk_size, (int)words_to_process.size());
            if (start_idx >= end_idx) continue;

            threads.emplace_back([=, &splits, &bpe_word_counts, &pair_stats, &stats_mutex, &words_to_process] {
                for (int w_idx = start_idx; w_idx < end_idx; ++w_idx) {
                    const std::string& word = words_to_process[w_idx];
                    auto& symbols = splits.at(word);
                    if (symbols.size() < 2) continue;
                    
                    int freq = bpe_word_counts.at(word);
                    std::vector<std::string> new_symbols;
                    new_symbols.reserve(symbols.size());

                    size_t k = 0;
                    while (k < symbols.size()) {
                        if (k < symbols.size() - 1 && symbols[k] == best_pair.first && symbols[k+1] == best_pair.second) {
                            // The merge happens here. We need to update stats for surrounding pairs.
                            std::lock_guard<std::mutex> lock(stats_mutex);

                            // Decrement count for the pair being destroyed to the left (if it exists)
                            // Example: if merging (b,c) in a,b,c -> destroy (a,b)
                            if (k > 0) {
                                pair_stats[{symbols[k-1], best_pair.first}] -= freq;
                            }
                            // Decrement count for the pair being destroyed to the right (if it exists)
                            // Example: if merging (b,c) in b,c,d -> destroy (c,d)
                            if (k < symbols.size() - 2) {
                                pair_stats[{best_pair.second, symbols[k+2]}] -= freq;
                            }

                            // Increment count for the new pair on the left
                            // Example: if merging (b,c) in a,b,c -> create (a,bc)
                            if (k > 0) {
                                pair_stats[{symbols[k-1], new_token}] += freq;
                            }
                            // Increment count for the new pair on the right
                            // Example: if merging (b,c) in b,c,d -> create (bc,d)
                            if (k < symbols.size() - 2) {
                                pair_stats[{new_token, symbols[k+2]}] += freq;
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
            });
        }
        for (auto& t : threads) t.join();

        // --- Step 3d: Log progress ---
        if ((i + 1) % 1000 == 0 || i == num_merges - 1) { // Print progress less frequently
            std::cout << "Merge " << i + 1 << "/" << num_merges << ": Merged '" << best_pair.first 
                      << "' and '" << best_pair.second << "' (Frequency: " << best_pair_freq << ")" << std::endl;
        }
    }

    // 4. FINALIZE (same as before)
    final_vocab.assign(vocab.begin(), vocab.end());
    std::sort(final_vocab.begin(), final_vocab.end(), [](const auto& a, const auto& b){ return a.length() > b.length(); });

    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
    std::cout << "BPE training complete. Final vocabulary size: " << this->vocSize << std::endl;
}

/**
 * @brief (NEW) Calculates final sub-token statistics from a pre-computed map of word counts.
 * This is orders of magnitude faster than iterating over all tokens in the corpus.
 *
 * @param corpus_word_counts Map of unique words and their frequencies.
 * @param outputPath Path to save the statistics CSV file.
 */
void tokeniser::calculateTokenStatsFromCounts(const std::map<std::string, int>& corpus_word_counts, const std::string& outputPath) {
    this->statOfEmbeddings.clear();

    auto is_word_for_bpe = [](const std::string& s) -> bool {
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0]));
    };

    // Iterate over unique words only
    for (const auto& pair : corpus_word_counts) {
        const std::string& pre_token = pair.first;
        const int count = pair.second;

        if (is_word_for_bpe(pre_token)) {
            std::vector<std::string> subwords;
            this->splitWord(pre_token, subwords); // Tokenize the unique word once
            for (const auto& subword : subwords) {
                // Add the total count of the original word to its sub-tokens
                this->statOfEmbeddings[subword] += count;
            }
        } else {
            // For punctuation/symbols, the token is the subword
            this->statOfEmbeddings[pre_token] += count;
        }
    }

    if (!outputPath.empty()) {
        std::cout << "-> Saving token statistics to: " << outputPath << std::endl;
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            std::cerr << "Warning: Could not open file to save token stats: " << outputPath << std::endl;
            return;
        }
        outFile << "token,repetitions\n";
        for (const auto& pair : this->statOfEmbeddings) {
            outFile << "\"" << pair.first << "\"," << pair.second << "\n";
        }
        outFile.close();
        std::cout << "-> Successfully saved statistics file." << std::endl;
    }
}
