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
#include <iomanip>
#include <set>
#include <unordered_set> 


// Create a global mutex and progress map for debugging.
std::mutex progress_mutex;
std::map<std::string, int> progress_map;

/**
 * @brief The main entry point for learning a BPE vocabulary.
 * This function now implements the standard BPE training pipeline.
 * It bypasses the complex pre-tokenization stage and directly feeds the
 * word counts from the corpus into the BPE refinement algorithm.
 * @param corpus_word_counts A map of unique words and their frequencies from the entire corpus.
 * @param num_merges The number of merge operations to perform.
 * @param final_vocab Output vector to store the learned vocabulary tokens (for external use/saving).
 */
void tokeniser::learn_vocabulary_from_word_counts(const std::map<std::string, int>& corpus_word_counts, int num_merges,
    std::vector<std::string>& final_vocab)
{
    // =========================================================================
    // REMOVED STAGE 1: The complex word segmentation was the source of the error.
    // We will now feed the raw word counts directly to the BPE algorithm.
    // =========================================================================
    std::cout << "[INFO] Starting BPE training directly from raw corpus word counts." << std::endl;
    std::cout << "[INFO] Total unique words for training: " << corpus_word_counts.size() << std::endl;

    // =========================================================================
    // STAGE 2 (NOW THE ONLY STAGE): BPE REFINEMENT
    // =========================================================================
    // Call the BPE algorithm with the original, complete word counts.
    // `groupCommonTokensParallel` will handle splitting words into characters.
    groupCommonTokensParallel(corpus_word_counts, num_merges, final_vocab);

    // =========================================================================
    // *** FINAL STATE ASSIGNMENT (This part is correct) ***
    // =========================================================================
    // The orchestrator function sets the final state of the tokenizer.
    this->tokens = final_vocab;
    this->vocSize = this->tokens.size();
}


/**
 * @brief Builds word counts with progress reporting tied to file completion.
 * This version uses the producer-consumer model and an event-driven logging
 * system. The main thread waits on a condition variable and prints a progress
 * update only when the producer thread signals that it has finished processing a file.
 * @param file_paths A vector of paths to the text files.
 * @param corpus_word_counts Output map to be filled with unique tokens and their total counts.
 */
// Now, replace the function itself
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

    // 2. DEFINE THE CONSUMER'S JOB
    auto consumer_task = [&work_queue]() -> std::unordered_map<std::string, int> {
        std::unordered_map<std::string, int> local_counts;
        std::vector<std::string> chunk;
        while (work_queue.wait_and_pop(chunk)) {
            for (const auto& line : chunk) {
                for (size_t i = 0; i < line.length(); ) {
                    unsigned char current_char = line[i];
                    if (std::isalpha(current_char)) {
                        size_t start = i;
                        while (i < line.length() && std::isalpha(static_cast<unsigned char>(line[i]))) i++;
                        // Extract the word with its original casing to allow for case-based splitting.
                        std::string original_word = line.substr(start, i - start);
                        
                        // Pre-split the word based on casing (e.g., camelCase).
                        std::vector<std::string> sub_words = pre_split_word(original_word);
                        
                        // Process each resulting sub-word.
                        for (auto& sub_word : sub_words) {
                            std::transform(sub_word.begin(), sub_word.end(), sub_word.begin(), [](unsigned char c){ return std::tolower(c); });
                            local_counts[sub_word]++;
                        }
                    } 
                    else if (!std::isspace(current_char)) {
                        local_counts[line.substr(i, 1)]++;
                        i++;
                    } 
                    else { i++; }
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

    // --- MODIFICATION: PRODUCER ---
    // The producer thread now updates the new 'last_file_completed' member.
    std::future<void> producer_future = std::async(std::launch::async, [&]() {
        for (const auto& path : file_paths) {
            std::string filename = std::filesystem::path(path).filename().string();
            std::ifstream file(path); // Open in text mode as before
            
            // --- MODIFIED LOGIC: Use stream position for accurate byte counting ---
            std::streampos last_pos = file.tellg(); // Get initial position (usually 0)

            std::string line;
            std::vector<std::string> chunk_buffer;
            chunk_buffer.reserve(CHUNK_SIZE);

            while (std::getline(file, line)) {
                chunk_buffer.push_back(std::move(line));
                if (chunk_buffer.size() >= CHUNK_SIZE) {
                    std::streampos current_pos = file.tellg();
                    long long bytes_in_chunk = current_pos - last_pos;
                    last_pos = current_pos;

                    {
                        std::lock_guard<std::mutex> lock(progress.mtx);
                        progress.bytes_read += bytes_in_chunk;
                    }
                    
                    work_queue.push(std::move(chunk_buffer));
                    chunk_buffer.clear();
                    chunk_buffer.reserve(CHUNK_SIZE);
                }
            }
            
            // After the loop, account for the final partial chunk
            file.clear(); // Clear EOF flags
            file.seekg(0, std::ios::end); // Seek to the end of the file
            std::streampos end_pos = file.tellg();
            long long final_bytes = end_pos - last_pos;

            if (!chunk_buffer.empty()) {
                work_queue.push(std::move(chunk_buffer));
            }

            // ATOMIC UPDATE AND SIGNAL
            {
                std::unique_lock<std::mutex> lock(progress.mtx);
                progress.bytes_read += final_bytes;
                progress.files_completed_count++;
                progress.last_file_completed = filename;
                progress.cv.notify_one();
            }
        }
        work_queue.close();
    });

    // 4. MAIN THREAD EVENT-DRIVEN LOGGING LOOP
    {
        std::unique_lock<std::mutex> lock(progress.mtx);
        size_t total_files = file_paths.size();
        
        // NEW: Local state for the main thread to track what has been reported.
        size_t last_reported_count = 0;

        // Loop until our local reported count matches the total number of files.
        while (last_reported_count < total_files) {
            
            // The predicate now checks if the shared count is greater than our local count.
            // The thread will now correctly sleep until the producer makes new progress.
            progress.cv.wait(lock, [&] { 
                return progress.files_completed_count > last_reported_count; 
            });

            // When we wake up, we know the count has increased. Print the new status.
            float percentage = 0.0;
            if (progress.total_bytes > 0) {
                percentage = static_cast<float>(progress.bytes_read) / progress.total_bytes * 100.0;
            }

            std::cout << "  -> Progress: [" << std::fixed << std::setprecision(4) << percentage << "%] "
                      << "\t| Completed " << progress.files_completed_count << "/" << total_files << " files. "
                      << "\t(Finished '" << progress.last_file_completed << "')" << std::endl;

            // Update our local state to match the new progress.
            last_reported_count = progress.files_completed_count;
        }
    }
    std::cout << "-> Producer has finished reading all files. Waiting for consumers...\n";

    // 5. AGGREGATE FINAL RESULTS (no changes needed here)
    std::unordered_map<std::string, int> final_counts;
    std::cout << "Counting: ";
    int i = 0;
    for (auto& f : consumer_futures) {
        i++;
        std::cout << i << " ";
        auto local_map = f.get();
        for (const auto& pair : local_map) {
            final_counts[pair.first] += pair.second;
        }
    }
    std::cout << std::endl;
    producer_future.get();
    corpus_word_counts.clear();
    corpus_word_counts.insert(final_counts.begin(), final_counts.end());
}



/**
 * @brief Calculates final sub-token statistics from a pre-computed map of word counts
 *        and saves them to a CSV file, sorted alphanumerically by token. This version 
 *        is orders of magnitude faster than iterating over all tokens in the corpus.
 * @param corpus_word_counts Map of unique words and their frequencies.
 * @param outputPath Path to save the statistics CSV file.
 */
void tokeniser::calculateTokenStatsFromCounts(const std::map<std::string, int>& corpus_word_counts, const std::string& outputPath) {
    this->statOfEmbeddings.clear();

    auto is_word_for_bpe = [](const std::string& s) -> bool {
        // A robust check for what constitutes a "word" split by BPE
        return !s.empty() && std::isalpha(static_cast<unsigned char>(s[0]));
    };

    std::cout << "Calculating final token statistics from " << corpus_word_counts.size() << " unique raw tokens..." << std::endl;
    // Iterate over unique words only
    int i = 0;
    for (const auto& pair : corpus_word_counts) {
        i++;
        if(i%10000 == 0) std::cout << i << " ";
        const std::string& pre_token = pair.first;
        const int count = pair.second;

        if (is_word_for_bpe(pre_token)) {
            std::vector<std::string> subwords;
            this->splitWord(pre_token, subwords); // Tokenize the unique word once
            for (const auto& subword : subwords) {
                // Add the total count of the original word to its sub-tokens
                this->statOfEmbeddings[subword] += count;
            }
        }
        else {
            // For punctuation/symbols, the token is the subword
            this->statOfEmbeddings[pre_token] += count;
        }
    }
    std::cout << std::endl;
    std::cout << "Calculation complete. Found " << this->statOfEmbeddings.size() << " final BPE tokens." << std::endl;

    if (!outputPath.empty()) {
        std::cout << "\n-> Sorting and saving token statistics to: " << outputPath << std::endl;
        
        // 1. Copy the map contents to a vector of pairs for sorting.
        std::vector<std::pair<std::string, int>> sorted_stats(
            this->statOfEmbeddings.begin(),
            this->statOfEmbeddings.end()
        );

        // 2. Sort the vector. The default sort for pairs will use the first element (the string)
        //    for comparison, which provides standard alphanumeric sorting.
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
            // Handle tokens that might contain commas or quotes
            std::string escaped_token;
            for (char c : token) {
                if (c == '"') escaped_token += "\"\"";
                else escaped_token += c;
            }
            outFile << "\"" << escaped_token << "\"," << pair.second << "\n";
        }
        outFile.close();
        std::cout << "-> Successfully saved sorted statistics file." << std::endl;
    }
    else {
        std::cout << "\nOutput path is empty. Skipped saving statistics file." << std::endl;
    }
}
