#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "include/tokenise.hpp"


// Function to read a single column CSV into a vector of strings
std::vector<std::string> readSingleColumnCsv(const std::string& filename) {
    std::vector<std::string> columnData;    // Vector to store the data
    std::ifstream file(filename);           // Open the file

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return columnData; // Return an empty vector on failure
    }

    std::string line;
    // Read each line from the file until the end
    while (std::getline(file, line)) {
        columnData.push_back(line);
    }

    file.close();
    return columnData;
}


// Function to read a specific column (0-based index) from a multi-column CSV
std::vector<std::string> readSpecificColumnFromCsv(const std::string& filename, int targetColumnIndex) {
    std::vector<std::string> columnData; // Vector to store the data from the target column
    std::ifstream file(filename);       // Open the file

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return columnData; // Return an empty vector on failure
    }

    std::string line;
    // Read each line from the file until the end
    while (std::getline(file, line)) {
        std::stringstream ss(line); // Create a stringstream from the current line
        std::string segment;        // To hold each comma-separated segment (column value)
        int currentColumnIndex = 0; // To keep track of the current column being processed
        bool columnFoundForThisLine = false; // Flag to check if the target column was found in the current line

        // Iterate through segments (columns) in the current line
        while (std::getline(ss, segment, ',')) {
            if (currentColumnIndex == targetColumnIndex) {
                columnData.push_back(segment); // Store the data from the target column
                columnFoundForThisLine = true;
                break; // We found our column for this line, no need to process further segments
            }
            currentColumnIndex++;
        }

        // If the target column was not found in the current line (e.g., line was too short)
        // Add an empty string to maintain the correct number of rows in columnData
        if (!columnFoundForThisLine) {
            columnData.push_back("");
        }
    }

    file.close(); // Close the file
    return columnData;
}


// Function to read an entire CSV file into a 2D vector of strings (row by row)
std::vector<std::vector<std::string>> readCsvTo2DVector(const std::string& filename) {
    std::vector<std::vector<std::string>> csvData; // The 2D vector to store all data
    std::ifstream file(filename);                   // Open the file

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        // Return an empty 2D vector on failure
        return csvData;
    }

    std::string line;
    // Read each line from the file until the end
    while (std::getline(file, line)) {
        std::vector<std::string> row;     // Vector to store the columns of the current row
        std::stringstream ss(line);       // Create a stringstream from the current line
        std::string segment;              // To hold each comma-separated segment (column value)

        // Iterate through segments (columns) in the current line
        // 'getline(ss, segment, ',')' extracts data until a comma is found
        while (std::getline(ss, segment, ',')) {
            row.push_back(segment); // Add the segment (column value) to the current row
        }

        // After parsing all segments in the line, add the completed row to the 2D vector
        csvData.push_back(row);
    }

    file.close(); // Close the file
    return csvData;
}


// Function to read a CSV with "word,count" format into an unordered_map<string, int>
std::unordered_map<std::string, int> readCorpusWordCount(const std::string& filename) {
    std::unordered_map<std::string, int> corpusWordCount;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return corpusWordCount;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string word_str;
        std::string count_str;

        // Extract the word (first column)
        if (std::getline(ss, word_str, ',')) {
            // Extract the count (second column)
            if (std::getline(ss, count_str)) { // Read rest of the line for count
                try {
                    int count = std::stoi(count_str);
                    corpusWordCount[word_str] = count;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid integer format for count in line: " << line << " - " << e.what() << std::endl;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Warning: Count out of range in line: " << line << " - " << e.what() << std::endl;
                }
            } else {
                std::cerr << "Warning: Missing count for word: " << word_str << " in line: " << line << std::endl;
            }
        } else {
            std::cerr << "Warning: Empty or malformed line (missing word): " << line << std::endl;
        }
    }

    file.close();
    return corpusWordCount;
}


// Function to read a CSV with "word,float1,float2,..." format into an unordered_map<string, vector<float>>
std::unordered_map<std::string, std::vector<float>> readMappedEmbeddings(const std::string& filename) {
    std::unordered_map<std::string, std::vector<float>> mappedEmbeddings;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return mappedEmbeddings;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string word_str;
        std::string segment;
        std::vector<float> embeddings_vector;

        // Extract the word (first column)
        if (std::getline(ss, word_str, ',')) {
            // Extract the rest of the floats
            while (std::getline(ss, segment, ',')) {
                try {
                    embeddings_vector.push_back(std::stof(segment));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid float format in line segment: '" << segment << "' in line: " << line << " - " << e.what() << std::endl;
                    // Optionally, you might want to clear the vector and skip this entry
                    embeddings_vector.clear();
                    break; // Stop processing this line
                } catch (const std::out_of_range& e) {
                    std::cerr << "Warning: Float out of range in line segment: '" << segment << "' in line: " << line << " - " << e.what() << std::endl;
                    embeddings_vector.clear();
                    break; // Stop processing this line
                }
            }
            // Only add to map if vector is not empty (i.e., no parsing errors for floats)
            if (!embeddings_vector.empty() || (line.find(',') != std::string::npos && embeddings_vector.empty())) { // Handle cases like "word," or "word"
                mappedEmbeddings[word_str] = embeddings_vector;
            } else if (line.find(',') == std::string::npos && !word_str.empty()){
                // Handle a line with just a word and no floats, if desired, add an empty vector
                 mappedEmbeddings[word_str] = {};
            }
        } else {
            std::cerr << "Warning: Empty or malformed line (missing word): " << line << std::endl;
        }
    }

    file.close();
    return mappedEmbeddings;
}
