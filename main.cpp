#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
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
        const int embeddingDimension = 64;      // embedding dimension
        const int d_val = 4;                    // Your original divisor for the formula
        const int num_merges = 24576;           // = 2^n and its multiples (24576 = 8192 * 3)
        const std::string path2folder = "D:/train/txt";
        // paths to all csv files
        const std::string unique_tokens_output_path = "D:/train/_unique_initial_tokens.csv";
        const std::string stats_output_path = "D:/train/_final_token_stats.csv";
        const std::string embeddings_output_path = "D:/train/_final_embeddings.csv";
        const std::string seed_output_path = "D:/train/_seedsForEmbeddings.csv";

        // Create and configure the tokenizer instance
    #ifdef USE_OPENCL
        tokeniser TOKENISER(embeddingDimension, d_val, ocl);
    #elif USE_CUDA || USE_CPU
        tokeniser TOKENISER(embeddingDimension, d_val);
    #endif

        TOKENISER.setNumThreads(); // Automatically set to use max threads
        std::cout << "-> Number of threads for CPU: " << TOKENISER.num_threads << std::endl;
        #ifdef USE_OPENCL
            TOKENISER.ocl = ocl;
        #endif

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
        std::unordered_map<std::string, int> corpus_word_counts;
        TOKENISER.buildCorpusWordCounts(all_file_paths, corpus_word_counts);
        std::cout << "-> Data aggregation complete. Total unique raw tokens: " << corpus_word_counts.size() << std::endl;
        if (corpus_word_counts.empty()) 
            throw std::runtime_error("No data loaded from files. Check file content.");

        // Step C: Save the gathered unique raw tokens to a CSV file.
        TOKENISER.saveUniqueTokensToCSV(corpus_word_counts, unique_tokens_output_path);

        std::cout << "---------------------- 2. VOCABULARY LEARNING ----------------------" << std::endl;
        std::vector<std::string> final_vocabulary;
        // Call the new two-stage learning function.
        TOKENISER.learn_vocabulary_from_word_counts(corpus_word_counts, num_merges, final_vocabulary);
        std::cout << "-> Vocabulary Learning complete. Final vocabulary size: " << TOKENISER.getVocabularySize() << std::endl;

        std::cout << "----------------- 3. STATS & EMBEDDING GEN ------------------" << std::endl;
        // Step A: Calculate statistics based on the final BPE vocabulary
        TOKENISER.calculateTokenStatsFromCounts(corpus_word_counts, stats_output_path);
        // Step B: Generate embeddings using original formula
        TOKENISER.generateAndSaveEmbeddings(embeddings_output_path, seed_output_path, -10.0f, 10.0f);

        std::cout << "--------------------- 4. INFERENCE DEMO ---------------------" << std::endl;
        std::string test_sentence = "This is a test sentence for christianity and its international relationships to see the new tokenizer in action. Hence, need more words to see whether it will work or not, if not rework the code logic and try again. This tokeniser is (BPE) is supercalifragilisticexpialidocious at the ludicrous speed. Ludicrous speed can be given by higher multiple of light speed which is 2.9 * 10^8 m/s.";
        std::vector<std::string> tokenized_sentence;
        TOKENISER.splitSentence(test_sentence, tokenized_sentence);
        std::cout << "Original: \"" << test_sentence << "\"" << std::endl;
        std::cout << "Tokenized: { ";
        for (const auto& token : tokenized_sentence) {
            std::cout << "'" << token << "' ";
        }
        std::cout << "}" << std::endl;
        std::cout << "Total tokens after tokenisation: " << tokenized_sentence.size() << std::endl;
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