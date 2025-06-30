#include <iostream>
#include <numeric>
#include <random>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <map> // Required for the new approach
#include "include/tokenise.hpp"

int main()
{
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "--------------------- Tokeniser 0.0.0.1 ---------------------" << std::endl;
    std::cout << "---------- Tokenisation based on BytePair Encoding ----------" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;

    #ifdef USE_CUDA
        std::cout << "Using CUDA" << std::endl;
    #elif USE_OPENCL
        std::cout << "Using OpenCL" << std::endl;
        OpenCLContext ocl; // This must be available if USE_OPENCL is defined
    #elif USE_CPU
        std::cout << "Using CPU core Functions" << std::endl;
    #endif

    try {
        // --- 1. CONFIGURATION ---
        const int embeddingDimension = 64;
        int d_val = 4;
        const int num_merges = 262144;
        const std::string path2folder = "D:/train/txt"; // data directory

        // Create the tokenizer instance
        tokeniser TOKENISER(embeddingDimension, d_val);
        TOKENISER.setNumThreads();
        std::cout << "-> Number of threads for CPU: " << TOKENISER.num_threads << std::endl;
        #ifdef USE_OPENCL
            TOKENISER.ocl = ocl;
        #endif

        // =========================================================================
        // MODIFICATION START: Using the efficient "word count" workflow
        // =========================================================================

        // --- 2. DATA AGGREGATION ---
        std::cout << "------------------- 1. AGGREGATING DATA ---------------------" << std::endl;
        
        // Step A: Collect all file paths first.
        std::vector<std::string> all_file_paths;
        for (const auto& entry : std::filesystem::directory_iterator(path2folder)) {
            if (entry.is_regular_file()) {
                all_file_paths.push_back(entry.path().string());
            }
        }
        std::cout << "-> Found " << all_file_paths.size() << " files for training in: " << path2folder << std::endl;

        if (all_file_paths.empty()) {
            throw std::runtime_error("No files found in the specified directory. Cannot proceed.");
        }
        
        // Step B: Build a map of unique words and their frequencies from all files in parallel.
        std::map<std::string, int> corpus_word_counts;
        TOKENISER.buildCorpusWordCountsParallel(all_file_paths, corpus_word_counts);
        
        std::cout << "-> Data aggregation complete. Total unique words and symbols found: " << corpus_word_counts.size() << std::endl;

        if (corpus_word_counts.empty()) {
            throw std::runtime_error("No data was loaded. Cannot proceed with training. Check your file content.");
        }

        // --- 3. BPE TRAINING ---
        std::cout << "---------------------- 2. BPE TRAINING ----------------------" << std::endl;
        std::cout << "-> Starting BPE training with " << num_merges << " merges..." << std::endl;
        
        std::vector<std::string> final_vocabulary;
        // Call the new overloaded function that takes the word counts map. This is much faster.
        TOKENISER.groupCommonTokensParallel(corpus_word_counts, num_merges, final_vocabulary);

        std::cout << "-> Training complete." << std::endl;
        std::cout << "-> Final vocabulary size: " << TOKENISER.getVocabularySize() << std::endl;

        // --- Calculate final token statistics ---
        std::cout << "-> Calculating final token statistics..." << std::endl;

        std::string stats_output_path = "D:/train/stats.csv";
        /*std::cout << "Enter the path to save the token statistics CSV (e.g., stats.csv), or press Enter to skip: ";
        std::getline(std::cin, stats_output_path);*/

        // Call the new, efficient statistics function.
        TOKENISER.calculateTokenStatsFromCounts(corpus_word_counts, stats_output_path);

        std::cout << "-> Statistics calculation complete." << std::endl;
        if (stats_output_path.empty()) {
            std::cout << "-> Skipped saving statistics file." << std::endl;
        }

        // =========================================================================
        // MODIFICATION END: The rest of the code works without changes.
        // =========================================================================

        // Optional: Display some of the token stats
        std::cout << "-> Top 100 most frequent tokens:" << std::endl;
        auto stats = TOKENISER.getTokenStats();
        std::vector<std::pair<std::string, int>> sorted_stats(stats.begin(), stats.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for(size_t i = 0; i < 100 && i < sorted_stats.size(); ++i) {
            std::cout << "  - Token: '" << sorted_stats[i].first << "', Count: " << sorted_stats[i].second << std::endl;
        }

        // --- 4. EMBEDDING GENERATION AND SAVING ---
        std::cout << "--------------- 3. GENERATE & SAVE EMBEDDINGS ---------------" << std::endl;

        std::string embeddings_output_path;
        std::cout << "Enter the path to save the token embeddings file (e.g., embeddings.csv): ";
        std::getline(std::cin, embeddings_output_path);

        if (embeddings_output_path.empty()) {
            embeddings_output_path = "D:/train/embeddings.csv"; // Default filename
            std::cout << "No path entered. Using default: " << embeddings_output_path << std::endl;
        }

        // Generate embeddings for the new vocabulary and save them to the specified file
        // TOKENISER.generateAndSaveEmbeddings(embeddings_output_path, -10.0f, 10.0f);

        // --- 5. (OPTIONAL) INFERENCE DEMO ---
        std::cout << "--------------------- 4. INFERENCE DEMO ---------------------" << std::endl;
        std::string test_sentence = "This is a test sentence to see the new tokenizer in action. Hence, need more words to see whether it will work or not, if not rework the code logic and try again.";
        std::vector<std::string> tokenized_sentence;
        TOKENISER.splitSentence(test_sentence, tokenized_sentence);
        
        std::cout << "Original: \"" << test_sentence << "\"" << std::endl;
        std::cout << "Tokenized: {";
        for (const auto& token : tokenized_sentence) {
            std::cout << "'" << token << "' ";
        }
        std::cout << "}" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        std::cout << "------------------------ PROCESS FAILED ------------------------" << std::endl;
        return 1;
    }

    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "----------------------- PROCESS COMPLETE --------------------" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    return 0;
}