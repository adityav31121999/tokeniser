#include <iostream>
#include <numeric>
#include <random>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <map>
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
        OpenCLContext ocl;
    #elif USE_CPU
        std::cout << "Using CPU core Functions" << std::endl;
    #endif

    try {
        // =========================================================================
        // --- 1. CONFIGURATION ---
        // =========================================================================
        const int embeddingDimension = 64;      // embedding dimension
        const int d_val = 4;                    // Your original divisor for the formula
        // number of merges: 2^n * m * 32768
        const int num_merges = 262144;          // 32768 * 8 * 1
        const std::string path2folder = "D:/train/txt";
        const std::string unique_tokens_output_path = "D:/train/_unique_initial_tokens.csv";
        const std::string stats_output_path = "D:/train/_final_token_stats.csv";
        const std::string embeddings_output_path = "D:/train/_final_embeddings.csv";
        // Create and configure the tokenizer instance
        tokeniser TOKENISER(embeddingDimension, d_val);
        TOKENISER.setNumThreads(); // Automatically set to use max threads
        
        std::cout << "-> Number of threads for CPU: " << TOKENISER.num_threads << std::endl;
        #ifdef USE_OPENCL
            TOKENISER.ocl = ocl;
        #endif

        // =========================================================================
        // --- 2. DATA AGGREGATION & INITIAL VOCABULARY SAVING ---
        // =========================================================================
        std::cout << "------------------- 1. AGGREGATING DATA ---------------------" << std::endl;
        
        // Step A: Collect all file paths
        std::vector<std::string> all_file_paths;
        for (const auto& entry : std::filesystem::directory_iterator(path2folder)) {
            if (entry.is_regular_file()) {
                all_file_paths.push_back(entry.path().string());
            }
        }
        std::cout << "-> Found " << all_file_paths.size() << " files for training in: " << path2folder << std::endl;
        if (all_file_paths.empty()) throw std::runtime_error("No files found in the specified directory.");

        // Step B: Build word counts using the robust producer-consumer model
        std::map<std::string, int> corpus_word_counts;
        TOKENISER.buildCorpusWordCountsParallel(all_file_paths, corpus_word_counts);
        std::cout << "-> Data aggregation complete. Total unique raw tokens: " << corpus_word_counts.size() << std::endl;
        if (corpus_word_counts.empty()) 
            throw std::runtime_error("No data loaded from files. Check file content.");

        // Step C: Save the gathered unique raw tokens to a CSV file.
        TOKENISER.saveUniqueTokensToCSV(corpus_word_counts, unique_tokens_output_path);

        // =========================================================================
        // --- 3. VOCABULARY LEARNING ---
        // =========================================================================
        std::cout << "---------------------- 2. VOCABULARY LEARNING ----------------------" << std::endl;
        
        std::vector<std::string> final_vocabulary;
        // Call the new two-stage learning function.
        TOKENISER.learn_vocabulary_from_word_counts(corpus_word_counts, num_merges, final_vocabulary);
        // -------------------------------------------------------------- //

        std::cout << "-> Vocabulary Learning complete. Final vocabulary size: " << TOKENISER.getVocabularySize() << std::endl;

        // =========================================================================
        // --- 4. FINAL STATISTICS AND EMBEDDING GENERATION ---
        // =========================================================================
        std::cout << "----------------- 3. STATS & EMBEDDING GEN ------------------" << std::endl;
        
        // Step A: Calculate statistics based on the final BPE vocabulary
        TOKENISER.calculateTokenStatsFromCounts(corpus_word_counts, stats_output_path);
        // Step B: Generate embeddings using original formula
        TOKENISER.generateAndSaveEmbeddings(embeddings_output_path, -10.0f, 10.0f);

        // =========================================================================
        // --- 5. (OPTIONAL) INFERENCE DEMO ---
        // =========================================================================
        std::cout << "--------------------- 4. INFERENCE DEMO ---------------------" << std::endl;
        
        std::string test_sentence = "This is a test sentence for christianity and its international relationships to see the new tokenizer in action. Hence, need more words to see whether it will work or not, if not rework the code logic and try again.";
        std::vector<std::string> tokenized_sentence;
        TOKENISER.splitSentence(test_sentence, tokenized_sentence);
        
        std::cout << "Original: \"" << test_sentence << "\"" << std::endl;
        std::cout << "Tokenized: { ";
        for (const auto& token : tokenized_sentence) {
            std::cout << "'" << token << "' ";
        }
        std::cout << "}" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        std::cout << "------------------------ PROCESS FAILED ------------------------" << std::endl;
        return 1;
    }


    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "----------------------- PROCESS COMPLETE --------------------" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    return 0;
}

/*
        const std::string t1 = "D:/train/_terminator_less1.csv";
        const std::string t2 = "D:/train/_terminator_less2.csv";
        const std::string t3 = "D:/train/_terminator_less3.csv";
        const std::string t4 = "D:/train/_terminator_less4.csv";
        const std::string f1 = "D:\\train\\TinyStories-train.txt";
        const std::string f2 = "D:\\train\\TinyStoriesV2-GPT4-train.txt";
        const std::string f3 = "D:\\train\\TinyStoriesV2-GPT4-valid.txt";
        const std::string f4 = "D:\\train\\TinyStories-valid.txt";
        const std::string terminator = "<|endoftext|>";

        // =========================================================================
        // --- 2. DATA AGGREGATION & INITIAL VOCABULARY SAVING ---
        // =========================================================================
        std::cout << "------------------- 1. AGGREGATING DATA ---------------------" << std::endl;
        
        // Step A: Collect all file paths
        std::vector<std::string> all_file_paths;
        for (const auto& entry : std::filesystem::directory_iterator(path2folder)) {
            if (entry.is_regular_file()) {
                all_file_paths.push_back(entry.path().string());
            }
        }
        std::cout << "-> Found " << all_file_paths.size() << " files for training in: " << path2folder << std::endl;
        if (all_file_paths.empty()) throw std::runtime_error("No files found in the specified directory.");
        
        // Step B: Build word counts using the robust producer-consumer model
        std::map<std::string, int> corpus_word_counts;
        TOKENISER.buildCorpusWordCountsParallel(all_file_paths, corpus_word_counts);
        std::cout << "-> Data aggregation complete. Total unique raw tokens: " << corpus_word_counts.size() << std::endl;
        if (corpus_word_counts.empty()) throw std::runtime_error("No data loaded from files. Check file content.");

        // Step C: Save the gathered unique raw tokens to a CSV file.
        TOKENISER.saveUniqueTokensToCSV(corpus_word_counts, unique_tokens_output_path);

        // =========================================================================
        // --- 3. BPE TRAINING ---
        // =========================================================================
        std::cout << "---------------------- 2. BPE TRAINING ----------------------" << std::endl;
        
        std::vector<std::string> final_vocabulary;
        // Use the highly optimized incremental update / inverted index algorithm
        TOKENISER.groupCommonTokensParallel(corpus_word_counts, num_merges, final_vocabulary);

        std::cout << "-> BPE Training complete. Final vocabulary size: " << TOKENISER.getVocabularySize() << std::endl;

        // =========================================================================
        // --- 4. FINAL STATISTICS AND EMBEDDING GENERATION ---
        // =========================================================================
        std::cout << "----------------- 3. STATS & EMBEDDING GEN ------------------" << std::endl;
        
        // Step A: Calculate statistics based on the final BPE vocabulary
        TOKENISER.calculateTokenStatsFromCounts(corpus_word_counts, stats_output_path);

        // Step B: Generate embeddings using original formula
        TOKENISER.generateAndSaveEmbeddings(embeddings_output_path, -10.0f, 10.0f);

        // =========================================================================
        // --- 5. (OPTIONAL) INFERENCE DEMO ---
        // =========================================================================
        std::cout << "--------------------- 4. INFERENCE DEMO ---------------------" << std::endl;
        
        std::string test_sentence = "This is a test sentence to see the new tokenizer in action. Hence, need more words to see whether it will work or not, if not rework the code logic and try again.";
        std::vector<std::string> tokenized_sentence;
        TOKENISER.splitSentence(test_sentence, tokenized_sentence);
        
        std::cout << "Original: \"" << test_sentence << "\"" << std::endl;
        std::cout << "Tokenized: { ";
        for (const auto& token : tokenized_sentence) {
            std::cout << "'" << token << "' ";
        }
        std::cout << "}" << std::endl;

*/