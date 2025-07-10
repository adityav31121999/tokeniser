# Tokeniser

A high-performance C++ implementation of a Byte Pair Encoding (BPE) tokenizer with custom embedding generation. This project is designed for speed, leveraging multi-threading for CPU-bound tasks and offering GPU acceleration via CUDA and OpenCL for numerical computations.

## Overview

This project takes a corpus of raw text files and performs the following key operations:

1.  **Corpus Processing**: It reads and pre-processes large volumes of text data in parallel, counting the frequency of each word.
2.  **Vocabulary Learning**: It uses the word counts to learn a subword vocabulary through an optimized Byte Pair Encoding (BPE) algorithm.
3.  **Embedding Generation**: It generates a unique, fixed-size numerical vector (embedding) for each token in the learned vocabulary using a custom mathematical formula.
4.  **Inference**: It provides the functionality to tokenize new sentences using the learned vocabulary.

The entire process is configurable and produces several CSV files containing the intermediate and final results.

## Features

-   **Optimized BPE Algorithm**: The BPE implementation in `group.cpp` uses an inverted index to track which words are affected by a merge. This avoids re-scanning the entire dataset on each merge, leading to a massive performance increase.
-   **Multi-threaded Corpus Processing**: The initial data aggregation (`corpus.cpp`) uses a highly efficient producer-consumer model. Producer threads read files and push chunks of lines into a thread-safe queue, while consumer threads process these chunks in parallel to build word counts.
-   **Parallel Map-Reduce**: Aggregating results from multiple threads is handled by a parallel merge tree (`merge.cpp`), which recursively combines maps for maximum efficiency.
-   **GPU Acceleration**: Embedding generation, a numerically intensive task, can be offloaded to the GPU. The project supports both:
    -   **CUDA**: Kernels in `kernel.cu` for NVIDIA GPUs.
    -   **OpenCL**: Kernels in `kernelcl.cpp` for broader compatibility with GPUs from different vendors (AMD, Intel, NVIDIA).
-   **Advanced Pre-processing**: Before BPE, words are intelligently pre-split (e.g., `camelCase` -> `camel`, `Case`) to provide a better starting point for the algorithm, as seen in `split.cpp`.
-   **Robust Progress Reporting**: The file processing pipeline provides detailed, real-time progress updates, including percentage completion and files processed.

## Workflow

### How BPE Works in This Project

The core of this project is its highly optimized implementation of the Byte Pair Encoding (BPE) algorithm. BPE learns a vocabulary by starting with single characters and iteratively merging the most frequent adjacent pairs of tokens. Here’s a step-by-step breakdown of the process as implemented in `group.cpp`:

1.  **Initial State: Character Splits**
    -   The process begins with a map of every unique word in the corpus and its frequency (`corpus_word_counts`).
    -   Punctuation and symbols are treated as atomic tokens and are immediately added to the vocabulary.
    -   Each word is split into its constituent characters, and a special end-of-word token `</w>` is appended. For example, the word `"hello"` (frequency: 50) becomes the sequence `['h', 'e', 'l', 'l', 'o', '</w']`.

2.  **The Inverted Index Optimization**
    -   Before any merges happen, the algorithm builds two crucial data structures just once:
        -   **Pair Frequencies**: A map that counts the total frequency of every adjacent pair of symbols. For `['h', 'e', 'l', 'l', 'o', '</w']` with a word frequency of 50, it would increment the counts for `('h', 'e')`, `('e', 'l')`, `('l', 'l')`, and `('l', 'o')` by 50.
        -   **Inverted Index**: A map from each pair to a list of the original words that contain it. For example, the pair `('l', 'l')` would point to a list like `["hello", "follow", ...]`.
    -   This initial setup is the key to the algorithm's speed.

3.  **The Merge Loop**
    -   The algorithm then iterates for a predefined number of merges (`num_merges`):
        1.  **Find Best Pair**: It finds the pair with the highest count in the frequency map. Let's say it's `('e', 's')`.
        2.  **Create New Token**: It merges this pair into a new token `"es"` and adds it to the vocabulary.
        3.  **Targeted Updates**: Instead of re-scanning the entire corpus, it uses the **inverted index** to get the short list of all words containing `('e', 's')` (e.g., `"test"`, `"less"`).
        4.  For *only these affected words*, it updates their token sequences (e.g., `['t', 'e', 's', 't', '</w']` becomes `['t', 'es', 't', '</w']`).
        5.  As it does this, it intelligently decrements the counts for the old, now-broken pairs (like `('t', 'e')` and `('s', 't')`) and increments the counts for the newly formed pairs (like `('t', 'es')`).
        6.  The merged pair `('e', 's')` is removed from the frequency map.

4.  **Final Vocabulary**
    -   After all merges are complete, the vocabulary consists of all the initial atomic tokens plus all the new subword tokens created during the merges.
    -   For inference, this vocabulary is sorted by token length in descending order, which is critical for the `splitWord` function.

5.  **Inference on New Text**
    -   When tokenizing a new word (e.g., `"testing"`), the `splitWord` function greedily finds the longest possible token from the vocabulary that matches the beginning of the word.
    -   Given a vocabulary sorted by length, it would match `"test"` before it matches `"t"`, ensuring an efficient and meaningful tokenization.

This inverted index approach avoids the quadratic complexity of naive BPE implementations, making it exceptionally fast even on very large vocabularies and corpora.

#### A Deeper Dive into the `groupCommonTokens` Implementation

The `groupCommonTokens` function in `group.cpp` is the heart of the vocabulary learning process. Here is a more granular, step-by-step walkthrough of its internal logic:

**Stage 1: Initialization and Pre-processing**

1.  **Segregation of Tokens**: The function first receives the `corpus_word_counts` map. It iterates through this map once to separate the tokens into two groups:
    *   **BPE Candidates**: Words that are suitable for being broken down (e.g., "hello", "testing"). These are stored in a temporary `bpe_word_counts` map.
    *   **Atomic Tokens**: Tokens that will not be split further. This includes all punctuation (`.`, `,`, `!`), symbols (`$`, `%`), and single-character words (`a`, `I`). These are inserted directly into the final `vocab` set.

2.  **Initial Character Splits**: For every BPE candidate word, the function performs an initial split into its constituent characters. A special end-of-word marker, `</w>`, is appended to each sequence. This marker is crucial for the model to distinguish between a token at the end of a word and the same token appearing in the middle of another word (e.g., "er" in "tester" vs. "er</w>" in "test er").
    *   `"hello"` (frequency 50) -> `splits["hello"] = {'h', 'e', 'l', 'l', 'o', '</w'}`
    *   All individual characters (`'h'`, `'e'`, etc.) and the `</w>` marker are also added to the `vocab`.

**Stage 2: Building the Inverted Index (The Core Optimization)**

This is the most critical step for performance. Instead of repeatedly scanning the word list, the algorithm builds two data structures *once* at the beginning:

1.  **Pair Frequencies (`pair_stats`)**: A map where the key is a pair of adjacent tokens (e.g., `('h', 'e')`) and the value is its total frequency across the entire corpus. This is calculated by iterating through the initial character splits and adding the parent word's frequency to the count for each pair.

2.  **Inverted Index (`inverted_index`)**: A map where the key is a token pair and the value is a list of all the original words that contain this pair.
    *   `inverted_index[('l', 'l')]` -> `{"hello", "follow", "dollar", ...}`

This initial investment in building the index allows for incredibly fast updates during the merge phase.

**Stage 3: The High-Speed Merge Loop**

The function then enters a loop that runs for the specified `num_merges`. In each iteration:

1.  **Find Best Pair**: It scans the `pair_stats` map to find the pair with the highest frequency. Let's say it's `('e', 's')` with a frequency of 5000.

2.  **Create New Token**: A new token is created by concatenating the pair (`"es"`) and is added to the `vocab`.

3.  **Targeted Updates**: This is where the inverted index shines.
    *   The algorithm looks up `('e', 's')` in the `inverted_index` to get a small, targeted list of all words that need to be updated (e.g., `{"test", "less", "address"}`).
    *   It then iterates *only* through this short list. For each affected word (e.g., `"test"`), it updates its token sequence: `{'t', 'e', 's', 't', '</w'}` becomes `{'t', "es", 't', '</w'}`.
    *   **Crucially**, as it performs this replacement, it also updates the `pair_stats` map in real-time:
        *   **Decrement**: The counts for pairs that were broken by the merge are decremented. For our example, the counts for `('t', 'e')` and `('s', 't')` would each be reduced by the frequency of the word "test".
        *   **Increment**: The counts for newly formed pairs are incremented. The count for `('t', "es")` would be increased by the frequency of "test".
    *   The `inverted_index` is also updated to reflect these new pairs.

4.  **Cleanup**: The now-merged pair `('e', 's')` is removed from `pair_stats` and `inverted_index` to prevent it from being selected again.

**Stage 4: Finalization**

1.  **Final Vocabulary**: After the loop completes, the `vocab` set contains all the initial atomic tokens plus all the new subword tokens created during the merges.
2.  **Sorting for Inference**: The contents of the `vocab` set are copied to the `final_vocab` vector and sorted by token length in **descending order**. This is essential for the `splitWord` function, which uses a greedy approach to tokenize new text. By sorting from longest to shortest, it ensures that it will match `"testing"` before it matches `"test"` or `"t"`, leading to the most efficient tokenization.

This inverted index strategy transforms the BPE algorithm from a process that gets slower with each merge into one that maintains high speed throughout, making it suitable for very large datasets and vocabularies.

### Multi-threaded Corpus Processing (The `buildCorpusWordCounts` function)

To rapidly process large text corpora, the project employs a highly efficient producer-consumer architecture. This design separates the I/O-bound task of reading files from the CPU-bound task of processing text, allowing both to run in parallel for maximum throughput.

1.  **Producers: The Readers**
    -   One or more "producer" threads are responsible for reading the `.txt` files from disk.
    -   Instead of processing lines one-by-one, a producer reads a large chunk of lines (e.g., 10,000) into a buffer.
    -   This entire chunk is then pushed into a central, thread-safe work queue.
    -   This approach maximizes I/O efficiency by reading large, contiguous blocks of data.

2.  **Consumers: The Processors**
    -   Multiple "consumer" threads run concurrently, waiting for work to appear in the queue.
    -   When a chunk of lines is available, a consumer pulls it from the queue and begins processing.
    -   For each line, it performs the necessary pre-processing: splitting words (e.g., `camelCase` -> `camel`, `Case`), converting to lowercase, and counting the frequency of each resulting token.
    -   Crucially, each consumer maintains its own **local** word count map. This avoids the massive performance bottleneck of having many threads trying to lock and update a single global map simultaneously.

3.  **Synchronization: The Thread-Safe Queue**
    -   A custom `ThreadSafeQueue` class acts as the backbone of this system. It uses a `std::mutex` to protect its internal state and a `std::condition_variable` to efficiently signal waiting consumers when new work is available or when all work is done. This avoids wasteful "busy-waiting".

4.  **Final Aggregation: The Parallel Merge Tree**
    -   After all files have been read and all consumers have finished, the result is a collection of local word count maps (one from each consumer).
    -   To combine these into a single master count, the project uses a parallel merge tree (`merge.cpp`). Instead of merging them sequentially (`map1 + map2 + map3...`), it merges pairs of maps in parallel (`(map1+map2)`, `(map3+map4)`, ...), then merges the results of those merges, and so on. This recursive, parallel approach significantly speeds up the final aggregation step.

5.  **Event-Driven Progress Reporting**
    -   The main thread doesn't waste cycles polling for progress. Instead, it waits on a `std::condition_variable`.
    -   A producer thread sends a signal every time it finishes reading a file, waking up the main thread just long enough to print a real-time progress update.

The `main.cpp` file orchestrates the entire tokenization pipeline:

1.  **Configuration**: Set parameters like embedding dimension, number of BPE merges, and data paths.

2.  **Data Aggregation (`buildCorpusWordCounts`)**:
    -   Scans the input directory for all text files.
    -   Producer threads read files line-by-line and push them into a work queue.
    -   Consumer threads pop from the queue, pre-process the text (splitting words, lowercasing), and count word frequencies into local maps.
    -   The local maps are efficiently merged into a single global `corpus_word_counts` map.
    -   The initial unique tokens are saved to `_unique_initial_tokens.csv`.

3.  **Vocabulary Learning (`learn_vocabulary_from_word_counts`)**:
    -   The `groupCommonTokens` function is called with the corpus word counts.
    -   It initializes a vocabulary with all single characters and atomic tokens (punctuation, etc.).
    -   It iteratively finds the most frequent pair of adjacent tokens and merges them into a new token, adding it to the vocabulary.
    -   This process repeats for the specified number of `num_merges`.

4.  **Final Statistics (`calculateTokenStatsFromCounts`)**:
    -   After the final vocabulary is learned, this function tokenizes every word from the original corpus using the new vocabulary.
    -   It calculates the total frequency of each final BPE token.
    -   The results are sorted and saved to `_final_token_stats.csv`.

5.  **Embedding Generation (`generateAndSaveEmbeddings`)**:
    -   Generates a random seed for each token in the final vocabulary.
    -   Calculates a `d`-dimensional embedding vector for each token using the specified backend (CPU, CUDA, or OpenCL).
    -   The tokens and their corresponding embeddings are saved to `_final_embeddings.csv`.

6.  **Inference Demo**:
    -   A sample sentence is tokenized using the `splitSentence` function to demonstrate how the final tokenizer works on new text.

## How to Build and Run

### Prerequisites
-   A C++17 compatible compiler (GCC, Clang, MSVC)
-   CMake (version 3.15 or higher)
-   (Optional) NVIDIA CUDA Toolkit if building with `USE_CUDA=ON`.
-   (Optional) An OpenCL SDK and drivers if building with `USE_OPENCL=ON`.

### Building

The project uses CMake for building. You can select the desired computation backend using a CMake flag.

```bash
# Create a build directory
mkdir build
cd build

# Configure for CPU
cmake .. -DUSE_CPU=ON

# Or, configure for CUDA
cmake .. -DUSE_CUDA=ON

# Or, configure for OpenCL
cmake .. -DUSE_OPENCL=ON

# Build the project
cmake --build . --config Release
```

### Running

After building, an executable will be created in the `build` directory. Before running, ensure you modify the `path2folder` in `main.cpp` to point to your text corpus directory.

```bash
./tokeniser
```

The program will print its progress to the console and generate the output CSV files in the specified location.

## Code Structure

| File                      | Description                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `main.cpp`                | The main driver application that orchestrates the entire workflow.       |
| `include/tokenise.hpp`    | Header file defining the core `tokeniser` class and helper structures.   |
| `corpus.cpp`              | Implements the multi-threaded corpus reading and word counting.          |
| `group.cpp`               | Contains the high-performance BPE vocabulary learning algorithm.         |
| `pairstats.cpp`           | Calculates final statistics for the learned BPE tokens.                  |
| `embedding.cpp`           | Manages embedding generation, wrapping CPU, CUDA, and OpenCL calls.      |
| `kernel.cu`               | Contains the CUDA kernels for GPU-accelerated embedding calculation.     |
| `kernelcl.cpp`            | Contains the OpenCL host wrappers for the GPU kernels.                   |
| `split.cpp`               | Implements the tokenization logic for inference on new text.             |
| `merge.cpp`               | Provides helper functions for parallel map merging.                      |
| `set.cpp`, `tokenise.cpp` | Contain constructors, setters, and getters for the `tokeniser` class.    |
