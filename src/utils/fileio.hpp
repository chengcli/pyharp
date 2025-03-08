#pragma once

// C/C++
#include <iostream>
#include <map>
#include <string>
#include <vector>

//! test file existance
bool file_exists(std::string fname);

//! test a blank line
bool is_blank_line(char const* line);
bool is_blank_line(std::string const& line);

//! decomment a file
std::string decomment_file(std::string fname);

//! get number of columns in a data table
int get_num_cols(std::string fname, char c = ' ');

//! get number of columns in a data table
int get_num_cols_str(std::string str, char c = ' ');

//! get number of rows in a data table
int get_num_rows(std::string fname);

//! get number of rows in a data table
int get_num_rows_str(std::string str);

//! replace a character in a string
void replace_char(char* buf, char c_old, char c_new);

char* strip_line(char* line);
char* next_line(char* line, int num, FILE* stream);
void read_data_table(char const* fname, double** data, int* rows, int* cols);

using DataVector = std::map<std::string, std::vector<double>>;
DataVector read_data_vector(std::string fname);
