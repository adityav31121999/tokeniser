
#include "include/tokenise.hpp"
#include <future>
#include <fstream>
#include <sstream>
#include <regex>
#include <map>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <string_view>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>


/**
 * @brief Pre-tokenizes a single word based on common patterns like camelCase or PascalCase.
 * This version uses a fast, manual character-by-character scan instead of std::regex,
 * which significantly improves performance in multithreaded contexts by avoiding
 * repeated regex compilation and execution overhead.
 * It splits words at transitions from lowercase to uppercase letters, and also handles acronyms
 * like in "MyHTTPRequest" -> "My", "HTTP", "Request".
 * Note: This heuristic will not split already-lowercased concatenated words like
 * "earlychristianwritings". That task is left to the BPE algorithm itself.
 * @param word The word to potentially split (case-sensitive).
 * @return A vector of sub-words. If no split occurs, it returns a vector with the original word.
 */
std::vector<std::string> pre_split_word(const std::string& word) {
    if (word.empty()) {
        return {};
    }

    std::vector<std::string> subtokens;
    size_t start = 0;

    for (size_t i = 1; i < word.length(); ++i) {
        bool split = false;
        const unsigned char prev_char = word[i-1];
        const unsigned char curr_char = word[i];

        // Condition 1: Lower to Upper (e.g., "camel|Case")
        if (std::islower(prev_char) && std::isupper(curr_char)) {
            split = true;
        }
        // Condition 2: Acronym to Word (e.g., "HTTP|Request")
        // This detects a transition from an uppercase sequence to a new capitalized word.
        else if (i + 1 < word.length() && std::isupper(prev_char) && std::isupper(curr_char) && std::islower(static_cast<unsigned char>(word[i+1]))) {
            split = true;
        }

        if (split) {
            subtokens.push_back(word.substr(start, i - start));
            start = i;
        }
    }

    // Add the last or only token.
    subtokens.push_back(word.substr(start));
    return subtokens;
}


/**
 * @brief Segments a word into the most likely sequence of subwords.
 * This function implements the "Word Break" problem using dynamic programming. It finds the
 * segmentation of a word that maximizes the total log-probability of its parts, where the
 * probability of a part is estimated from its frequency in the entire corpus.
 * @param word The word to be segmented.
 * @param corpus_word_counts A map of all unique words in the corpus to their frequencies.
 * @return A vector of strings representing the optimal segmentation of the word.
 */
std::vector<std::string> pre_tokenize_word_by_corpus_freq(const std::string& word, const std::map<std::string, int>& corpus_word_counts) {
    const size_t n = word.length();
    if (n == 0) return {};

    // DP table: dp[i] stores the max score for a segmentation of the prefix word[0...i-1].
    std::vector<float> dp(n + 1, -std::numeric_limits<float>::infinity());
    // back_pointers[i] stores the length of the last subword in the optimal segmentation of word[0...i-1].
    std::vector<int> back_pointers(n + 1, 0);
    
    dp[0] = 0.0; // Base case: an empty prefix has a score of 0.

    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            const std::string subword = word.substr(j, i - j);
            auto it = corpus_word_counts.find(subword);

            if (it != corpus_word_counts.end()) {
                float score = std::log(static_cast<float>(it->second));
                
                if (dp[j] != -std::numeric_limits<float>::infinity() && dp[j] + score > dp[i]) {
                    dp[i] = dp[j] + score;
                    back_pointers[i] = i - j; // Store the length of this subword.
                }
            }
        }
    }

    if (dp[n] == -std::numeric_limits<float>::infinity()) {
        return {word}; // No valid segmentation found, return original word.
    }

    std::vector<std::string> segmentation;
    int current_pos = n;
    while (current_pos > 0) {
        int last_word_len = back_pointers[current_pos];
        segmentation.push_back(word.substr(current_pos - last_word_len, last_word_len));
        current_pos -= last_word_len;
    }
    std::reverse(segmentation.begin(), segmentation.end());
    
    return segmentation;
}


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
 * @brief Splits a single word into a sequence of subword tokens using the learned vocabulary.
 * MODIFIED: This function now only tokenizes the word's body. The caller is responsible
 * for adding the end-of-word marker.
 * @param word The word to be tokenized.
 * @param subwords Output vector to store the resulting subword tokens.
 */
void tokeniser::splitWord(const std::string& word, std::vector<std::string>& subwords) const {
    subwords.clear();
    if (word.empty()) return;

    // MODIFICATION: Process the word directly, without the </w> token.
    std::string current_word = word;
    
    while (!current_word.empty()) {
        bool found_match = false;
        // Greedily find the longest token in our vocabulary that is a prefix of the current word.
        // This requires `this->tokens` to be sorted by length, descending.
        for (const auto& token : this->tokens) {
            if (current_word.rfind(token, 0) == 0) { // check if string starts with token
                subwords.push_back(token);
                current_word = current_word.substr(token.length());
                found_match = true;
                break;
            }
        }
        if (!found_match) {
            // This fallback is critical if a character was never seen in training.
            // For example, if your vocabulary doesn't contain the token `</w>`, this
            // logic would split it into '<', '/', 'w', '>'.
            std::string unknown_char = current_word.substr(0, 1);
            subwords.push_back(unknown_char);
            current_word = current_word.substr(1);
        }
    }
}


/**
 * @brief Tokenizes a full sentence into a sequence of subword tokens.
 * MODIFIED: This function now correctly adds the </w> marker after each word is tokenized.
 * @param sentence The input sentence string.
 * @param all_subwords Output vector to store the final sequence of tokens.
 */
void tokeniser::splitSentence(const std::string& sentence, std::vector<std::string>& all_subwords) const {
    all_subwords.clear();

    // Regex to split sentence by words and punctuation.
    std::regex re(R"([a-zA-Z]+|[^a-zA-Z\s])");
    auto words_begin = std::sregex_iterator(sentence.begin(), sentence.end(), re);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::string token_str = i->str();
        
        // If it's a word (starts with a letter), split it using our learned vocabulary
        if (!token_str.empty() && std::isalpha(static_cast<unsigned char>(token_str[0]))) {
            std::string lower_token_str = token_str;
            std::transform(lower_token_str.begin(), lower_token_str.end(), lower_token_str.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            
            std::vector<std::string> word_subwords;
            splitWord(lower_token_str, word_subwords);
            
            // MODIFICATION: Add the end-of-word marker here, after tokenizing the word body.
            word_subwords.push_back("</w>");
            
            all_subwords.insert(all_subwords.end(), word_subwords.begin(), word_subwords.end());
        } 
        else {
            // It's punctuation or another symbol, keep it as a single token.
            all_subwords.push_back(token_str);
        }
    }
}


/**
 * @brief Extracts tokens from a large text file using multiple threads.
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


/**
 * @brief Saves all unique tokens (words and punctuation) to a single-column CSV file.
 * The keys from the provided map are used as the tokens.
 * @param corpus_word_counts The map containing all unique tokens as keys.
 * @param outputPath The path where the CSV file will be saved.
 */
void tokeniser::saveUniqueTokensToCSV(const std::map<std::string, int>& corpus_word_counts, const std::string& outputPath) {
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


/**
 * @brief Reads a file, breaks it into parts based on a terminator, and converts each part
 * into a single continuous line. This version is multi-threaded and processes document parts in parallel.
 * @param originalFile file with terminators
 * @param newFile new file with no terminator
 * @param terminator terminator character
 */
 void splitFileUsingTerminator(const std::string& originalFile, const std::string& newFile, const std::string& terminator) {
    // 1. Open input file and read its entire content into a string.
    // This approach is simple and robust for files that fit comfortably in memory.
    std::ifstream inFile(originalFile);
    if (!inFile.is_open()) {
        throw std::runtime_error("Error: Could not open original file: " + originalFile);
    }

    std::stringstream buffer;
    buffer << inFile.rdbuf();
    std::string content = buffer.str();
    inFile.close();

    if (content.empty()) {
        // Create an empty file and return if the input is empty.
        std::ofstream(newFile).close();
        return;
    }

    std::cout << "Process Started for: " << originalFile << std::endl;

    // 2. Find all document boundaries without processing them yet.
    std::vector<std::pair<size_t, size_t>> part_boundaries;
    size_t start_pos = 0;
    size_t end_pos;
    while ((end_pos = content.find(terminator, start_pos)) != std::string::npos) {
        part_boundaries.emplace_back(start_pos, end_pos - start_pos);
        start_pos = end_pos + terminator.length();
    }
    // Add the last part if it exists after the final terminator
    if (start_pos < content.length()) {
        part_boundaries.emplace_back(start_pos, content.length() - start_pos);
    }

    // Helper lambda to clean up a text block. It takes a non-owning string_view
    // and returns an owned, cleaned string.
    auto clean_part_from_view = [](std::string_view text_part_view) -> std::string {
        std::string text_part(text_part_view); // Create a modifiable string
        text_part = std::regex_replace(text_part, std::regex(R"(\r\n|\r|\n)"), " ");
        text_part = std::regex_replace(text_part, std::regex(R"(\s{2,})"), " ");
        text_part = std::regex_replace(text_part, std::regex(R"(^\s+|\s+$)"), "");
        return text_part;
    };

    // 3. Process parts in parallel using std::async.
    std::vector<std::future<std::string>> futures;
    std::string_view content_view(content);
    std::cout << "  -> Parallel Processing each part" << std::endl;
    for (const auto& boundary : part_boundaries) {
        futures.push_back(std::async(std::launch::async, [content_view, boundary, &clean_part_from_view]() {
            std::string_view part_view = content_view.substr(boundary.first, boundary.second);
            return clean_part_from_view(part_view);
        }));
    }

    // 4. Monitor progress and write results.
    std::atomic<size_t> completed_count = 0;
    std::atomic<bool> all_done = false;
    std::thread progress_thread;

    // Only start the progress thread if there's something to process.
    if (!futures.empty()) {
        progress_thread = std::thread([&]() {
            size_t total_parts = futures.size();
            // Print initial status
            std::cout << "  -> Progress for " << originalFile << ": "
                      << std::fixed << std::setprecision(2) << 0.00 << "% "
                      << "(0/" << total_parts << " documents processed)" << std::endl;

            while (!all_done.load()) {
                // This is a simple sleep-based timer.
                std::this_thread::sleep_for(std::chrono::seconds(30));
                
                if (all_done.load()) break; // Check again after waking up

                size_t completed = completed_count.load();
                double percentage = static_cast<double>(completed) / total_parts * 100.0;
                
                std::cout << "  -> Progress for " << originalFile << ": "
                          << std::fixed << std::setprecision(2) << percentage << "% "
                          << "(" << completed << "/" << total_parts << " documents processed)" << std::endl;
            }
        });
    }

    // 5. Open the output file and write the results as they become available.
    std::cout << "  -> New file created" << std::endl;
    std::ofstream outFile(newFile);
    if (!outFile.is_open()) {
        if (progress_thread.joinable()) {
            all_done = true;
            progress_thread.join();
        }
        throw std::runtime_error("Error: Could not open new file for writing: " + newFile);
    }

    size_t total_parts = futures.size();
    for (size_t i = 0; i < total_parts; ++i) {
        std::string cleaned_part = futures[i].get(); // This will wait for the future to be ready
        if (!cleaned_part.empty()) {
            outFile << cleaned_part << "\n";
        }
        completed_count++; // Atomically increment the counter
    }

    // 6. Clean up the progress thread.
    if (progress_thread.joinable()) {
        all_done = true;
        progress_thread.join();
    }

    // Print final progress report
    if (total_parts > 0) {
        std::cout << "  -> Progress for " << originalFile << ": "
                  << std::fixed << std::setprecision(2) << 100.00 << "% "
                  << "(" << total_parts << "/" << total_parts << " documents processed)" << std::endl;
    }
    
    outFile.close();
    std::cout << "Process Completed with: " << newFile << std::endl;
}
