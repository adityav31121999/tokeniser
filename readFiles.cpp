#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "include/tokenise.hpp"

// Utility function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Utility function to remove quotes from a string
std::string removeQuotes(const std::string& str) {
    std::string result = str;
    
    // Remove leading and trailing quotes (both double and single quotes)
    if (result.length() >= 2 && 
        ((result.front() == '"' && result.back() == '"') ||
         (result.front() == '\'' && result.back() == '\''))) {
        result = result.substr(1, result.length() - 2);
    }
    
    return result;
}

// Function to check if a line looks like a CSV header
bool isHeaderLine(const std::string& line) {
    std::string trimmed = trim(line);
    std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(), ::tolower);
    
    // Common header patterns - checks for typical CSV header combinations
    return (trimmed.find("token") != std::string::npos && 
            (trimmed.find("count") != std::string::npos || 
             trimmed.find("repetitions") != std::string::npos)) ||
           (trimmed.find("word") != std::string::npos && 
            trimmed.find("count") != std::string::npos) ||
           (trimmed.find("embedding") != std::string::npos) ||
           (trimmed == "word,count" || trimmed == "token,count" || 
            trimmed == "token,repetitions");
}

// Function to read a single column CSV into a vector of strings
std::vector<std::string> readSingleColumnCsv(const std::string& filename) {
    std::vector<std::string> columnData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return columnData;
    }

    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Trim whitespace and remove carriage returns
        line = trim(line);
        
        if (!line.empty()) {
            columnData.push_back(line);
        }
    }

    file.close();
    
    if (columnData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << columnData.size() 
                  << " entries from file " << filename << std::endl;
    }
    
    return columnData;
}

// Function to read a specific column (0-based index) from a multi-column CSV
std::vector<std::string> readSpecificColumnFromCsv(const std::string& filename, int targetColumnIndex) {
    std::vector<std::string> columnData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return columnData;
    }

    if (targetColumnIndex < 0) {
        std::cerr << "Error: Invalid column index " << targetColumnIndex 
                  << ". Column index must be non-negative." << std::endl;
        file.close();
        return columnData;
    }

    std::string line;
    int lineNumber = 0;
    bool headerSkipped = false;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Trim and handle different line endings
        line = trim(line);
        
        if (line.empty()) {
            continue;
        }
        
        // Skip header line
        if (!headerSkipped && isHeaderLine(line)) {
            headerSkipped = true;
            std::cout << "Skipping header line: " << line << std::endl;
            continue;
        }
        
        std::stringstream ss(line);
        std::string segment;
        int currentColumnIndex = 0;
        bool columnFoundForThisLine = false;

        while (std::getline(ss, segment, ',')) {
            segment = trim(segment);
            segment = removeQuotes(segment); // Remove quotes
            
            if (currentColumnIndex == targetColumnIndex) {
                columnData.push_back(segment);
                columnFoundForThisLine = true;
                break;
            }
            currentColumnIndex++;
        }

        if (!columnFoundForThisLine) {
            if (lineNumber <= 3) { // Only warn for first few lines to avoid spam
                std::cerr << "Warning: Column " << targetColumnIndex 
                          << " not found in line " << lineNumber << " of file " << filename 
                          << " (line has " << currentColumnIndex << " columns)" << std::endl;
            }
            columnData.push_back(""); // Add empty string to maintain row count
        }
    }

    file.close();
    
    if (columnData.empty()) {
        std::cerr << "Warning: No data found in column " << targetColumnIndex 
                  << " of file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << columnData.size() 
                  << " entries from column " << targetColumnIndex 
                  << " of file " << filename << std::endl;
    }
    
    return columnData;
}

// Function to read an entire CSV file into a 2D vector of strings (row by row)
std::vector<std::vector<std::string>> readCsvTo2DVector(const std::string& filename) {
    std::vector<std::vector<std::string>> csvData;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return csvData;
    }

    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        line = trim(line);
        
        if (line.empty()) {
            continue;
        }
        
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string segment;

        while (std::getline(ss, segment, ',')) {
            row.push_back(trim(segment)); // Trim each segment
        }

        if (!row.empty()) {
            csvData.push_back(row);
        }
    }

    file.close();
    
    if (csvData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << csvData.size() 
                  << " rows from file " << filename << std::endl;
    }
    
    return csvData;
}

// Function to read a CSV with "word,count" format into an unordered_map<string, int>
std::unordered_map<std::string, int> readUnorderedMap(const std::string& filename) {
    std::unordered_map<std::string, int> corpusWordCount;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return corpusWordCount;
    }

    std::string line;
    int lineNumber = 0;
    int successfullyParsed = 0;
    bool headerSkipped = false;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        line = trim(line);
        
        if (line.empty()) {
            continue;
        }
        
        // Skip header line
        if (!headerSkipped && isHeaderLine(line)) {
            headerSkipped = true;
            std::cout << "Skipping header line: " << line << std::endl;
            continue;
        }
        
        std::stringstream ss(line);
        std::string word_str;
        std::string count_str;

        // Extract the word (first column)
        if (std::getline(ss, word_str, ',')) {
            word_str = trim(word_str);
            word_str = removeQuotes(word_str);
            
            // Extract the count (second column)
            if (std::getline(ss, count_str)) {
                count_str = trim(count_str);
                count_str = removeQuotes(count_str);
                
                // Handle cases where count might have extra quotes or formatting
                // Remove any remaining quotes that might be embedded
                count_str.erase(std::remove(count_str.begin(), count_str.end(), '"'), count_str.end());
                count_str.erase(std::remove(count_str.begin(), count_str.end(), '\''), count_str.end());
                
                if (word_str.empty()) {
                    if (lineNumber <= 5) { // Limit warnings to first few lines
                        std::cerr << "Warning: Empty word in line " << lineNumber 
                                  << " of file " << filename << std::endl;
                    }
                    continue;
                }
                
                try {
                    int count = std::stoi(count_str);
                    corpusWordCount[word_str] = count;
                    successfullyParsed++;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid integer format for count '" 
                              << count_str << "' in line " << lineNumber 
                              << " of file " << filename << ": " << e.what() << std::endl;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Warning: Count out of range '" 
                              << count_str << "' in line " << lineNumber 
                              << " of file " << filename << ": " << e.what() << std::endl;
                }
            } else {
                std::cerr << "Warning: Missing count for word '" << word_str 
                          << "' in line " << lineNumber << " of file " << filename << std::endl;
            }
        } else {
            if (lineNumber <= 5) { // Limit warnings to first few lines
                std::cerr << "Warning: Empty or malformed line " << lineNumber 
                          << " in file " << filename << std::endl;
            }
        }
    }

    file.close();
    
    if (corpusWordCount.empty()) {
        std::cerr << "Warning: No valid word-count pairs found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << successfullyParsed 
                  << " word-count pairs from file " << filename << std::endl;
    }
    
    return corpusWordCount;
}

// Function to read a CSV with "word,float1,float2,..." format into an unordered_map<string, vector<float>>
std::unordered_map<std::string, std::vector<float>> readMappedEmbeddings(const std::string& filename) {
    std::unordered_map<std::string, std::vector<float>> mappedEmbeddings;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Check if the file exists and has proper read permissions." << std::endl;
        return mappedEmbeddings;
    }

    std::string line;
    int lineNumber = 0;
    int successfullyParsed = 0;
    int expectedEmbeddingSize = -1; // To track consistency
    bool headerSkipped = false;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        line = trim(line);
        
        if (line.empty()) {
            continue;
        }
        
        // Skip header line
        if (!headerSkipped && (isHeaderLine(line) || 
                               line.find("token") != std::string::npos ||
                               line.find("embedding") != std::string::npos)) {
            headerSkipped = true;
            std::cout << "Skipping header line: " << line << std::endl;
            continue;
        }
        
        std::stringstream ss(line);
        std::string word_str;
        std::string segment;
        std::vector<float> embeddings_vector;

        // Extract the word (first column)
        if (std::getline(ss, word_str, ',')) {
            word_str = trim(word_str);
            word_str = removeQuotes(word_str);
            
            if (word_str.empty()) {
                if (lineNumber <= 5) { // Limit warnings
                    std::cerr << "Warning: Empty word in line " << lineNumber 
                              << " of file " << filename << std::endl;
                }
                continue;
            }
            
            // Extract the rest of the floats
            bool parseError = false;
            while (std::getline(ss, segment, ',')) {
                segment = trim(segment);
                segment = removeQuotes(segment);
                
                if (segment.empty()) {
                    continue; // Skip empty segments
                }
                
                // Remove any remaining quotes that might be embedded
                segment.erase(std::remove(segment.begin(), segment.end(), '"'), segment.end());
                segment.erase(std::remove(segment.begin(), segment.end(), '\''), segment.end());
                
                try {
                    float value = std::stof(segment);
                    embeddings_vector.push_back(value);
                } catch (const std::invalid_argument& e) {
                    // Skip the problematic value and continue
                    if (lineNumber <= 5) { // Limit warnings to first few lines
                        std::cerr << "Warning: Invalid float format '" << segment 
                                  << "' for word '" << word_str << "' in line " << lineNumber 
                                  << " of file " << filename << ": " << e.what() << std::endl;
                    }
                    // Don't set parseError = true, just skip this value
                    continue;
                } catch (const std::out_of_range& e) {
                    if (lineNumber <= 5) {
                        std::cerr << "Warning: Float out of range '" << segment 
                                  << "' for word '" << word_str << "' in line " << lineNumber 
                                  << " of file " << filename << ": " << e.what() << std::endl;
                    }
                    continue;
                }
            }
            
            // Only add to map if we got at least some embeddings
            if (!embeddings_vector.empty()) {
                // Check embedding size consistency
                if (expectedEmbeddingSize == -1) {
                    expectedEmbeddingSize = embeddings_vector.size();
                } else if (embeddings_vector.size() != expectedEmbeddingSize) {
                    if (lineNumber <= 5) { // Limit warnings
                        std::cerr << "Warning: Inconsistent embedding size for word '" 
                                  << word_str << "' in line " << lineNumber 
                                  << " of file " << filename 
                                  << " (expected " << expectedEmbeddingSize 
                                  << ", got " << embeddings_vector.size() << ")" << std::endl;
                    }
                }
                
                mappedEmbeddings[word_str] = embeddings_vector;
                successfullyParsed++;
            } else {
                // Handle tokens with no embeddings - add empty vector
                mappedEmbeddings[word_str] = {};
                successfullyParsed++;
            }
        } else {
            if (lineNumber <= 5) { // Limit warnings
                std::cerr << "Warning: Empty or malformed line " << lineNumber 
                          << " in file " << filename << std::endl;
            }
        }
    }

    file.close();
    
    if (mappedEmbeddings.empty()) {
        std::cerr << "Warning: No valid word-embedding pairs found in file " << filename << std::endl;
    } else {
        std::cout << "Successfully read " << successfullyParsed 
                  << " word-embedding pairs from file " << filename;
        if (expectedEmbeddingSize != -1) {
            std::cout << " (embedding dimension: " << expectedEmbeddingSize << ")";
        }
        std::cout << std::endl;
    }
    
    return mappedEmbeddings;
}