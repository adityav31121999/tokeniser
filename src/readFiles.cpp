#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "include/tokenise.hpp"


// Helper function to read a single CSV field, handling quotes and escaped quotes
// Modified to return the raw field as read, without trimming or quote removal.
// This allows the caller to decide how to process the field.
std::string readCsvField(std::stringstream& ss) {
    std::string field = ""; // Initialize field to an empty string
    char c;

    // Consume leading whitespace before the field starts (if any)
    while (ss.peek() != EOF && std::isspace(static_cast<unsigned char>(ss.peek()))) {
        ss.get();
    }

    if (ss.peek() == '"') { // If the field starts with a quote...
        ss.get(); // Consume opening quote
        bool in_quotes = true;
        while (ss.get(c)) { // Loop until the end of the quoted field or stream
            if (c == '"') {
                if (ss.peek() == '"') { // Escaped double quote ("")
                    field += '"';
                    ss.get(); // Consume the second quote
                } else { // Closing quote
                    in_quotes = false;
                    break; // Exit the loop for this field
                }
            } else {
                field += c;
            }
        }
        // After finding the closing quote, consume any trailing whitespace then the comma
        while (ss.peek() != EOF && std::isspace(static_cast<unsigned char>(ss.peek()))) {
            ss.get();
        }
        if (ss.peek() == ',') { // Consume the comma only if it exists
            ss.get(); // Consume the comma delimiter
        }
    } else { // If the field is not quoted...
        // Not a quoted field, read until comma or end of line
        std::getline(ss, field, ',');
    }

    // After reading, check if the field is still empty and if it's the first field (indicated by the starting position of the stream).
    // If so, return an empty string to signal a potential issue with the CSV format.
    if (field.empty() && ss.tellg() == std::streampos(0)) {
        return ""; // Indicate an empty field at the start of the line
    }

    return field; // Return the field as read
}




// Function to read an entire CSV file into a 2D vector of floats
std::vector<std::vector<float>> readCsvTo2DVector(const std::string& filename) {
    std::vector<std::vector<float>> csvData;
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
        if (line.empty()) { // Check for truly empty lines before trimming
            continue; // Skip truly empty lines
        }
        // Apply global trim to the entire line
        line = trim(line);
        if (line.empty()) {
            continue; // Skip lines that become empty after trimming
        }

        std::vector<float> row;
        std::stringstream ss(line);

        // Loop to read fields using the robust readCsvField
        // Check for success of readCsvField and if there are actual characters processed
        while (ss.good()) { // Continue as long as the stream is good (not failed)
            std::string raw_field = readCsvField(ss);
            if (raw_field.empty() && ss.tellg() != std::streampos(0)) { // Skip if empty field and not at the start of line
                // Check if it's the start of the line and the field is empty
                continue;
            }

            // Apply trim and removeQuotes to the raw field string before conversion
            std::string cleaned_field_str = removeQuotes(trim(raw_field));

            float value = 0.0f;
            // Attempt to convert the string to float
            try {
                // If the cleaned_field_str is empty, stof will throw invalid_argument, which is desired.
                value = std::stof(cleaned_field_str);
            }
            catch (const std::invalid_argument& e) {
                // Conversion failed (e.g., empty string, "abc" to float)
                // This means the field was empty or non-numeric. Treat as 0.0f.
                if (!cleaned_field_str.empty()) { // Only warn if it wasn't just an empty field
                    std::cerr << "Warning: Failed to convert field '" << cleaned_field_str
                              << "' to float at line " << lineNumber
                              << " in file " << filename << ". Defaulting to 0.0f." << std::endl;
                }
                value = 0.0f;
            }
            catch (const std::out_of_range& e) {
                std::cerr << "Warning: Float out of range '" << cleaned_field_str
                          << "' at line " << lineNumber
                          << " in file " << filename << ". Defaulting to 0.0f." << std::endl;
                value = 0.0f;
            }
            row.push_back(value);
        } // End of while(has_more_fields_on_line)

        // Add row only if it's not empty (e.g., not just an empty line that resulted in no fields)
        if (!row.empty() || line.find(',') != std::string::npos) { // Add row even if empty but it indicates fields (e.g. "a,,c")
            csvData.push_back(row);
        }
    }

    file.close();
    if (csvData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    }
    else {
        std::cout << "Successfully read " << csvData.size()
                  << " rows from file " << filename << std::endl;
    }

    return csvData;
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
        if (line.empty()) {
            continue;
        }

        line = trim(line); // Trim whitespace from the entire line
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string raw_field = readCsvField(ss); // Use the new helper for robustness
        columnData.push_back(removeQuotes(trim(raw_field))); // Process the extracted field
    }

    file.close();
    if (columnData.empty()) {
        std::cerr << "Warning: No data found in file " << filename << std::endl;
    }
    else {
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

    while (std::getline(file, line)) {
        lineNumber++;

        if (line.empty()) {
            continue;
        }

        line = trim(line);

        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string segment_raw;
        int currentColumnIndex = 0;
        bool columnFoundForThisLine = false;

        bool has_more_fields_on_line = true;
        while (has_more_fields_on_line) {
            segment_raw = readCsvField(ss);
            
            if (segment_raw.empty() && ss.eof()) { // Check for empty field at end of line
                 has_more_fields_on_line = false;
            }

            if (currentColumnIndex == targetColumnIndex) {
                columnData.push_back(removeQuotes(trim(segment_raw)));
                columnFoundForThisLine = true;
                break; // Found the target column, move to next line
            }
            currentColumnIndex++;

            if (ss.peek() == EOF && !ss.good()) { // Check if stream failed or reached true EOF
                has_more_fields_on_line = false;
            }
        }

        if (!columnFoundForThisLine) {
            // Only warn if the line genuinely had fewer columns than expected,
            // not if it was just a malformed last line.
            if (currentColumnIndex <= targetColumnIndex) { // Check if we even reached the target column index
                if (lineNumber <= 10) { // Limit warnings for brevity
                    std::cerr << "Warning: Column " << targetColumnIndex
                              << " not found in line " << lineNumber << " of file " << filename
                              << " (line has only " << currentColumnIndex << " columns)." << std::endl;
                }
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
    bool headerSkipped = false; // Keep this for potential future use or if some files truly have headers

    while (std::getline(file, line)) {
        lineNumber++;
        if (line.empty()) {
            continue;
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        
        // Read the token field using the robust helper
        std::string raw_token_field = readCsvField(ss);
        std::string token = removeQuotes(trim(raw_token_field));
        // Read the count field using the robust helper
        std::string raw_count_field = readCsvField(ss);
        std::string count_str_cleaned = removeQuotes(trim(raw_count_field));

        // Attempt to convert to integer
        try {
            // std::stoi is quite robust for numeric strings, including leading/trailing spaces
            // and signs. It stops at the first non-numeric character.
            int count = std::stoi(count_str_cleaned);
            corpusWordCount[token] = count; // Allow empty string as a key if it comes from data
            successfullyParsed++;
        }
        catch (const std::invalid_argument& e) {
            if (!count_str_cleaned.empty() && count_str_cleaned.find_first_not_of(" \t\r\n") != std::string::npos) { // Only warn if it wasn't just empty/whitespace
                std::cerr << "Warning: Invalid integer format for count '"
                          << raw_count_field << "' (cleaned to '" << count_str_cleaned << "') in line " << lineNumber
                          << " of file " << filename << ": " << e.what() << ". Skipping entry." << std::endl;
            }
        }
        catch (const std::out_of_range& e) {
            std::cerr << "Warning: Count out of range '"
                      << raw_count_field << "' (cleaned to '" << count_str_cleaned << "') in line " << lineNumber
                      << " of file " << filename << ": " << e.what() << ". Skipping entry." << std::endl;
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


// Your tokeniser::readFromFiles method remains largely the same,
// as it calls the updated readUnorderedMap and readMappedEmbeddings functions.
void tokeniser::readFromFiles(const std::string& path2ClassDataFolder) 
{
    std::string token_stats_file = path2ClassDataFolder + "/_final_token_stats.csv";
    // Add robust file existence check
    if (!std::filesystem::exists(token_stats_file)) {
        std::cerr << "Error: Token statistics file not found at " << token_stats_file << std::endl;
        throw std::runtime_error("Required token statistics file missing. Ensure training created '_final_token_stats.csv' in the specified path.");
    }
    this->statOfTokens = readUnorderedMap(token_stats_file);
    this->vocSize = count_lines(token_stats_file);

    this->tokens.clear(); // Ensure it's empty before populating
    this->tokens.resize(this->vocSize); // Pre-allocate space
    tokens = readSpecificColumnFromCsv(token_stats_file, 0);

    this->embeddings.clear(); // Clear existing embeddings
    std::vector<std::vector<float>> embeddings1 = readCsvTo2DVector(path2ClassDataFolder + "/_embeddings_only.csv");
    this->embeddings = std::move(embeddings1);

    // Update vocabulary size based on loaded data
    this->d = this->embeddings[0].size();
    this->vocSize = this->tokens.size();

    //Sort tokens by length in descending order
    std::sort(this->tokens.begin(), this->tokens.end(), 
        [](const auto& a, const auto& b) { 
            return a.length() > b.length(); 
        });

    std::cout << "Tokenizer initialized successfully:" << std::endl;
    std::cout << "  - Tokens loaded: " << this->tokens.size() << std::endl;
    std::cout << "  - Vocabulary size: " << this->vocSize << std::endl;
    std::cout << "  - Embedding dimension: " << this->d << std::endl;
    // std::cout << "  - deEmbedding count: " << this->deEmbeddings.size() << std::endl;
}
