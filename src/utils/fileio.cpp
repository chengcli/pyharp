// C/C++ header
#include <cctype>  // isspace
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// utils header
#include "fileio.hpp"

bool file_exists(std::string fname) {
  std::ifstream ifile(fname.c_str());
  return ifile.is_open();
}

bool is_blank_line(char const* line) {
  for (char const* cp = line; *cp; ++cp) {
    if (!std::isspace(*cp)) return false;
  }
  return true;
}

bool is_blank_line(std::string const& line) {
  return is_blank_line(line.c_str());
}

std::string decomment_file(std::string fname) {
  std::stringstream msg;
  if (!file_exists(fname)) {
    throw std::runtime_error("decomment_file: file not found: " + fname);
  }

  std::ifstream file(fname.c_str(), std::ios::in);
  std::string ss;
  char c;
  while (file) {
    file.get(c);
    if (c == '#') {
      while (c != '\n' && file) file.get(c);
      continue;
    }
    ss += c;
  }
  return ss;
}

int get_num_cols(std::string fname, char c) {
  std::ifstream inp(fname.c_str(), std::ios::in);
  std::string line;
  std::getline(inp, line);
  if (line.empty()) return 0;
  int cols = line[0] == c ? 0 : 1;

  for (int i = 1; i < line.length(); ++i)
    if (line[i - 1] == c && line[i] != c) cols++;
  return cols;
}

int get_num_cols_str(std::string str, char c) {
  std::string line;
  std::stringstream ss(str);
  std::getline(ss, line);
  if (line.empty()) return 0;
  int cols = line[0] == c ? 0 : 1;

  for (int i = 1; i < line.length(); ++i)
    if (line[i - 1] == c && line[i] != c) cols++;
  return cols;
}

int get_num_rows(std::string fname) {
  std::ifstream inp(fname.c_str(), std::ios::in);
  std::string line;
  int rows = 0;

  while (std::getline(inp, line)) ++rows;
  return rows;
}

int get_num_rows_str(std::string str) {
  std::string line;
  int rows = 0;
  std::stringstream ss(str);

  while (std::getline(ss, line)) ++rows;
  return rows;
}

void replace_char(char* buf, char c_old, char c_new) {
  int len = strlen(buf);
  for (int i = 0; i < len; ++i)
    if (buf[i] == c_old) buf[i] = c_new;
}

char* strip_line(char* line) {
  char* p = line;
  int len = strlen(line);
  // strip newline or carriage rtn
  while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
    line[--len] = 0;
  // advance to first non-whitespace
  while (isspace(*p)) p++;
  // advance to first non-whitespace
  // skip characters aftet '#'
  char* pp = p;
  while (*pp != '#' && *pp) pp++;
  *pp = 0;
  return p;
}

char* next_line(char* line, int num, FILE* stream) {
  char* p;
  while (fgets(line, num, stream) != NULL) {
    p = strip_line(line);
    if (strlen(p) > 0) break;
  }
  return p;
}

DataVector read_data_vector(std::string fname) {
  DataVector amap;

  std::ifstream input(fname.c_str(), std::ios::in);
  std::stringstream ss, msg;
  std::string line, sbuffer;
  std::vector<std::string> field;

  if (!input.is_open()) {
    throw std::runtime_error("read_data_vector: file not found: " + fname);
  }

  getline(input, line);
  ss.str(line);
  while (!ss.eof()) {
    ss >> sbuffer;
    field.push_back(sbuffer);
  }
  ss.clear();

  while (getline(input, line)) {
    if (line.empty()) continue;
    ss.str(line);
    for (std::vector<std::string>::iterator f = field.begin(); f != field.end();
         ++f) {
      double value;
      ss >> value;
      amap[*f].push_back(value);
    }
    ss.clear();
  }

  return amap;
}
