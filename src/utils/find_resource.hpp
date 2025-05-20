#pragma once

// C/C++
#include <string>
#include <vector>

namespace harp {
extern char search_paths[65536];
extern char pathsep;

//! Strip non-printing characters wherever they are
/*!
 * \ingroup resource
 *
 * \param s        Input string
 * \returns a copy of the string, stripped of all non- printing characters.
 */
std::string stripnonprint(std::string const& s);

//! Serialize search paths.
/*!
 * \ingroup resource
 *
 * \param dirs  Vector of strings containing the directories to be serialized
 * \return  pointer to string containing the serialized directories
 *
 */
char* serialize_search_paths(std::vector<std::string> const& dirs);

//! Deserialize search paths.
/*!
 * \ingroup resource
 *
 * \param p Pointer to string containing the serialized directories
 * \return  Vector of strings containing the deserialized directories
 *
 */
std::vector<std::string> deserialize_search_paths(char const* p);

//! Set the default directories for input files.
/*!
 * \ingroup resource
 *
 * Searches for input files along a path that includes platform-
 * specific default locations, and possibly user-specified locations.
 * This function installs the platform-specific directories on the search
 * path. It is invoked at startup by appinit(), and never should need to
 * be called by user programs.
 *
 * The current directory (".") is always searched first.
 *
 * Additional directories may be added by calling function
 * add_resource_directory.
 */
void set_default_directories();

//! Add a directory to the data file search path.
/*!
 * \ingroup resource
 *
 * \param dir  String name for the directory to be added to the search path
 *
 */
void add_resource_directory(std::string const& dir, bool prepend = true);

//! Find a resource file.
/*!
 * \ingroup resource
 *
 * This routine will search for a file in the default locations specified
 * for the application. See the routine setDefaultDirectories() listed
 * above. The first directory searched is usually the current working
 * directory.
 *
 * The default set of directories will not be searched if an absolute path
 * (for example, one starting with `/` or `C:\`) or a path relative to the
 * user's home directory (for example, starting with `~/`) is specified.
 *
 * The presence of the file is determined by whether the file can be
 * opened for reading by the current user.
 *
 * \param name Name of the input file to be searched for
 * \return  The absolute path name of the first matching file
 *
 * If the file is not found an exception is thrown.
 *
 */
std::string find_resource(std::string const& name);
}  // namespace harp
