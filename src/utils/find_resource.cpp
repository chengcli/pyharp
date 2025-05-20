// C/C++
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>

// harp
#include <configure.h>

#include "find_resource.hpp"

namespace harp {
// const char* search_paths = "";
char search_paths[65536] = ".";
static std::mutex dir_mutex;

#ifdef WINDOWS
char pathsep = ';';
#else
char pathsep = ':';
#endif

std::string stripnonprint(std::string const& s) {
  std::string ss = "";
  for (size_t i = 0; i < s.size(); i++) {
    if (isprint(s[i])) {
      ss += s[i];
    }
  }
  return ss;
}

char* serialize_search_paths(std::vector<std::string> const& dirs) {
  std::string s = "";
  for (size_t i = 0; i < dirs.size(); i++) {
    s += dirs[i];
    if (i + 1 < dirs.size()) {
      s += pathsep;
    }
  }
  strncpy(search_paths, s.c_str(), 65536);
  return search_paths;
}

std::vector<std::string> deserialize_search_paths(char const* p) {
  std::vector<std::string> dirs;
  std::string s(p);
  size_t start = 0;
  size_t end = s.find(pathsep);
  while (end != std::string::npos) {
    dirs.push_back(s.substr(start, end - start));
    start = end + 1;
    end = s.find(pathsep, start);
  }
  dirs.push_back(s.substr(start, end));
  bool found_root = false;
  for (auto dir : dirs) {
    if (dir.find(HARP_ROOT_DIR) != std::string::npos) {
      found_root = true;
      break;
    }
  }
  if (!found_root) {
    dirs.push_back(std::string(HARP_ROOT_DIR) + "/data");
  }
  return dirs;
}

void set_default_directories() {
  std::vector<std::string> input_dirs;

  // always look in the local directory first
  input_dirs.push_back(".");

  // add cia data directory
  input_dirs.push_back(std::string(HARP_ROOT_DIR) + "/python/cia_legacy");

  serialize_search_paths(input_dirs);
}

void add_resource_directory(std::string const& dir, bool prepend) {
  std::unique_lock<std::mutex> dirLock(dir_mutex);
  auto input_dirs = deserialize_search_paths(search_paths);
  std::string d = stripnonprint(dir);

  // Expand "~/" to user's home directory, if possible
  if (d.find("~/") == 0 || d.find("~\\") == 0) {
    char* home = getenv("HOME");  // POSIX systems
    if (!home) {
      home = getenv("USERPROFILE");  // Windows systems
    }
    if (home) {
      d = home + d.substr(1, std::string::npos);
    }
  }

  // Remove any existing entry for this directory
  auto iter = std::find(input_dirs.begin(), input_dirs.end(), d);
  if (iter != input_dirs.end()) {
    input_dirs.erase(iter);
  }

  if (prepend) {
    // Insert this directory at the beginning of the search path
    input_dirs.insert(input_dirs.begin(), d);
  } else {
    // Append this directory to the end of the search path
    input_dirs.push_back(d);
  }

  serialize_search_paths(input_dirs);
}

std::string find_resource(std::string const& name) {
  std::unique_lock<std::mutex> dirLock(dir_mutex);
  std::string::size_type islash = name.find('/');
  std::string::size_type ibslash = name.find('\\');
  std::string::size_type icolon = name.find(':');

  std::vector<std::string> dirs = deserialize_search_paths(search_paths);

  // Expand "~/" to user's home directory, if possible
  if (name.find("~/") == 0 || name.find("~\\") == 0) {
    char* home = getenv("HOME");  // POSIX systems
    if (!home) {
      home = getenv("USERPROFILE");  // Windows systems
    }
    if (home) {
      std::string full_name = home + name.substr(1, std::string::npos);
      std::ifstream fin(full_name);
      if (fin) {
        return full_name;
      } else {
        std::string msg = "\nharp::find_resource::" + name + "not found";
        throw std::runtime_error(msg.c_str());
      }
    }
  }

  // If this is an absolute path, just look for the file there
  if (islash == 0 || ibslash == 0 ||
      (icolon == 1 && (ibslash == 2 || islash == 2))) {
    std::ifstream fin(name);
    if (fin) {
      return name;
    } else {
      std::string msg = "\nharp::find_resource::" + name + "not found";
      throw std::runtime_error(msg.c_str());
    }
  }

  // Search the data directories for the input file, and return
  // the full path if a match is found
  size_t nd_ = dirs.size();
  for (size_t i = 0; i < nd_; i++) {
    std::string full_name = dirs[i] + "/" + name;
    std::ifstream fin(full_name);
    if (fin) {
      return full_name;
    }
  }
  std::string msg = "\nResource " + name + " not found in director";
  msg += (nd_ == 1 ? "y " : "ies ");
  for (size_t i = 0; i < nd_; i++) {
    msg += "\n'" + dirs[i] + "'";
    if (i + 1 < nd_) {
      msg += ", ";
    }
  }
  msg += "\n\n";
  msg += "To fix this problem, either:\n";
  msg += "    a) move the missing files into the local directory;\n";
  msg += "    b) define -DMYPATH= during build\n";
  throw std::runtime_error(msg);
}
}  // namespace harp
