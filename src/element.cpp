// C/C++
#include <map>
#include <sstream>
#include <vector>

// harp
#include "constants.h"
#include "element.hpp"
#include "utils/strings.hpp"

namespace harp {

/**
 * Database for atomic weights.
 *
 * If no value is given in either source, it is because no stable isotopes of
 * that element are known and the atomic weight of that element is listed here
 * as -1.0
 *
 * units = kg / kg-mol (or equivalently gm / gm-mol)
 */
struct AtomicWeightData {
  std::string symbol;    //!< Element symbol, first letter capitalized
  std::string fullName;  //!< Element full name, first letter lowercase
  double atomicWeight;   //!< Element atomic weight in kg / kg-mol, if known. -1
                         //!< if no stable isotope
};

/**
 * Database for named isotopic weights.
 * Values are used from Kim, et al. @cite kim2019.
 *
 * units = kg / kg-mol (or equivalently gm / gm-mol)
 *
 * This structure was picked because it's simple, compact, and extensible.
 */
struct IsotopeWeightData {
  std::string symbol;    //!< Isotope symbol, first letter capitalized
  std::string fullName;  //!< Isotope full name, first letter lowercase
  double atomicWeight;   //!< Isotope atomic weight in kg / kg-mol
  int atomicNumber;      //!< Isotope atomic number
};

/**
 * @var static vector<atomicWeightData> AtomicWeightTable
 * @brief AtomicWeightTable is a vector containing the atomic weights database.
 *
 * AtomicWeightTable is a static variable with scope limited to this file.
 * It can only be referenced via the functions in this file.
 *
 * The size of the table is given by the initial instantiation.
 */
static std::vector<AtomicWeightData> AtomicWeightTable{
    {"H", "hydrogen", 1.008},
    {"He", "helium", 4.002602},
    {"Li", "lithium", 6.94},
    {"Be", "beryllium", 9.0121831},
    {"B", "boron", 10.81},
    {"C", "carbon", 12.011},
    {"N", "nitrogen", 14.007},
    {"O", "oxygen", 15.999},
    {"F", "fluorine", 18.998403163},
    {"Ne", "neon", 20.1797},
    {"Na", "sodium", 22.98976928},
    {"Mg", "magnesium", 24.305},
    {"Al", "aluminum", 26.9815384},
    {"Si", "silicon", 28.085},
    {"P", "phosphorus", 30.973761998},
    {"S", "sulfur", 32.06},
    {"Cl", "chlorine", 35.45},
    {"Ar", "argon", 39.95},
    {"K", "potassium", 39.0983},
    {"Ca", "calcium", 40.078},
    {"Sc", "scandium", 44.955908},
    {"Ti", "titanium", 47.867},
    {"V", "vanadium", 50.9415},
    {"Cr", "chromium", 51.9961},
    {"Mn", "manganese", 54.938043},
    {"Fe", "iron", 55.845},
    {"Co", "cobalt", 58.933194},
    {"Ni", "nickel", 58.6934},
    {"Cu", "copper", 63.546},
    {"Zn", "zinc", 65.38},
    {"Ga", "gallium", 69.723},
    {"Ge", "germanium", 72.630},
    {"As", "arsenic", 74.921595},
    {"Se", "selenium", 78.971},
    {"Br", "bromine", 79.904},
    {"Kr", "krypton", 83.798},
    {"Rb", "rubidium", 85.4678},
    {"Sr", "strontium", 87.62},
    {"Y", "yttrium", 88.90584},
    {"Zr", "zirconium", 91.224},
    {"Nb", "nobelium", 92.90637},
    {"Mo", "molybdenum", 95.95},
    {"Tc", "technetium", -1.0},
    {"Ru", "ruthenium", 101.07},
    {"Rh", "rhodium", 102.90549},
    {"Pd", "palladium", 106.42},
    {"Ag", "silver", 107.8682},
    {"Cd", "cadmium", 112.414},
    {"In", "indium", 114.818},
    {"Sn", "tin", 118.710},
    {"Sb", "antimony", 121.760},
    {"Te", "tellurium", 127.60},
    {"I", "iodine", 126.90447},
    {"Xe", "xenon", 131.293},
    {"Cs", "cesium", 132.90545196},
    {"Ba", "barium", 137.327},
    {"La", "lanthanum", 138.90547},
    {"Ce", "cerium", 140.116},
    {"Pr", "praseodymium", 140.90766},
    {"Nd", "neodymium", 144.242},
    {"Pm", "promethium", -1.0},
    {"Sm", "samarium", 150.36},
    {"Eu", "europium", 151.964},
    {"Gd", "gadolinium", 157.25},
    {"Tb", "terbium", 158.925354},
    {"Dy", "dysprosium", 162.500},
    {"Ho", "holmium", 164.930328},
    {"Er", "erbium", 167.259},
    {"Tm", "thulium", 168.934218},
    {"Yb", "ytterbium", 173.045},
    {"Lu", "lutetium", 174.9668},
    {"Hf", "hafnium", 178.49},
    {"Ta", "tantalum", 180.94788},
    {"W", "tungsten", 183.84},
    {"Re", "rhenium", 186.207},
    {"Os", "osmium", 190.23},
    {"Ir", "iridium", 192.217},
    {"Pt", "platinum", 195.084},
    {"Au", "gold", 196.966570},
    {"Hg", "mercury", 200.592},
    {"Tl", "thallium", 204.38},
    {"Pb", "lead", 207.2},
    {"Bi", "bismuth", 208.98040},
    {"Po", "polonium", -1.0},
    {"At", "astatine", -1.0},
    {"Rn", "radon", -1.0},
    {"Fr", "francium", -1.0},
    {"Ra", "radium", -1.0},
    {"Ac", "actinium", -1.0},
    {"Th", "thorium", 232.0377},
    {"Pa", "protactinium", 231.03588},
    {"U", "uranium", 238.02891},
    {"Np", "neptunium", -1.0},
    {"Pu", "plutonium", -1.0},
    {"Am", "americium", -1.0},
    {"Cm", "curium", -1.0},
    {"Bk", "berkelium", -1.0},
    {"Cf", "californium", -1.0},
    {"Es", "einsteinium", -1.0},
    {"Fm", "fermium", -1.0},
    {"Md", "mendelevium", -1.0},
    {"No", "nobelium", -1.0},
    {"Lr", "lawrencium", -1.0},
    {"Rf", "rutherfordium", -1.0},
    {"Db", "dubnium", -1.0},
    {"Sg", "seaborgium", -1.0},
    {"Bh", "bohrium", -1.0},
    {"Hs", "hassium", -1.0},
    {"Mt", "meitnerium", -1.0},
    {"Ds", "darmstadtium", -1.0},
    {"Rg", "roentgenium", -1.0},
    {"Cn", "copernicium", -1.0},
    {"Nh", "nihonium", -1.0},
    {"Gl", "flerovium", -1.0},
    {"Mc", "moscovium", -1.0},
    {"Lv", "livermorium", -1.0},
    {"Ts", "tennessine", -1.0},
    {"Og", "oganesson", -1.0},
};

/**
 * @var static vector<IsotopeWeightData> IsotopeWeightTable
 * @brief IsotopeWeightTable is a vector containing the atomic weights database.
 *
 * IsotopeWeightTable is a static variable with scope limited to this file.
 * It can only be referenced via the functions in this file.
 *
 * The size of the table is given by the initial instantiation.
 */
static std::vector<IsotopeWeightData> IsotopeWeightTable{
    // M. Wang et al. The AME2016 atomic mass evaluation. Chinese Physics C.
    // doi:10.1088/1674-1137/41/3/030003.
    {"D", "deuterium", 2.0141017781, 1},
    {"Tr", "tritium", 3.0160492820, 1},
    {"E", "electron", constants::ElectronMass* constants::Avogadro, 0},
};

// This is implemented as a separate function from elementSymbols() because this
// pattern allows elementSymbols() to return a const reference to the data.
std::vector<std::string> elementVectorsFromSymbols() {
  std::vector<std::string> values;
  for (const auto& atom : AtomicWeightTable) {
    values.push_back(atom.symbol);
  }
  return values;
}

const std::vector<std::string>& element_symbols() {
  const static std::vector<std::string> values = elementVectorsFromSymbols();
  return values;
}

// This is implemented as a separate function from elementNames() because this
// pattern allows elementNames() to return a const reference to the data.
std::vector<std::string> elementVectorsFromNames() {
  std::vector<std::string> values;
  for (const auto& atom : AtomicWeightTable) {
    values.push_back(atom.fullName);
  }
  return values;
}

const std::vector<std::string>& element_names() {
  const static std::vector<std::string> values = elementVectorsFromNames();
  return values;
}

std::map<std::string, double> mapAtomicWeights() {
  std::map<std::string, double> symMap;

  for (auto const& atom : AtomicWeightTable) {
    symMap.emplace(atom.symbol, atom.atomicWeight);
    symMap.emplace(atom.fullName, atom.atomicWeight);
  }
  for (auto const& isotope : IsotopeWeightTable) {
    symMap.emplace(isotope.symbol, isotope.atomicWeight);
    symMap.emplace(isotope.fullName, isotope.atomicWeight);
  }
  return symMap;
}

const std::map<std::string, double>& element_weights() {
  const static std::map<std::string, double> symMap = mapAtomicWeights();
  return symMap;
}

double get_element_weight(const std::string& ename) {
  const auto& elementMap = element_weights();
  double elementWeight = 0.0;
  std::string symbol = trim_copy(ename);
  auto search = elementMap.find(symbol);
  if (search != elementMap.end()) {
    elementWeight = search->second;
  } else {
    std::string name = to_lower_copy(symbol);
    search = elementMap.find(name);
    if (search != elementMap.end()) {
      elementWeight = search->second;
    }
  }
  if (elementWeight > 0.0) {
    return elementWeight;
  } else if (elementWeight < 0.0) {
    throw std::runtime_error("At get_element_weight: element '" + ename +
                             "' has no stable isotopes");
  }
  throw std::runtime_error("At get_element_weight: element not found: '" +
                           ename + "'");
}

double get_element_weight(int atomicNumber) {
  int num = static_cast<int>(num_elements_defined());
  if (atomicNumber > num || atomicNumber < 1) {
    throw std::runtime_error(
        "At get_element_weight: AtomicWeightTable index out of bounds: " +
        std::to_string(atomicNumber) +
        " while num_elements_defined() = " + std::to_string(num));
  }
  double elementWeight = AtomicWeightTable[atomicNumber - 1].atomicWeight;
  if (elementWeight < 0.0) {
    throw std::runtime_error("At get_element_weight: element '" +
                             get_element_name(atomicNumber) +
                             "' has no stable isotopes");
  }
  return elementWeight;
}

std::string get_element_symbol(const std::string& ename) {
  std::string name = to_lower_copy(trim_copy(ename));
  for (const auto& atom : AtomicWeightTable) {
    if (name == atom.fullName) {
      return atom.symbol;
    }
  }
  for (const auto& atom : IsotopeWeightTable) {
    if (name == atom.fullName) {
      return atom.symbol;
    }
  }
  throw std::runtime_error("At getElementSymbol: element not found: '" + ename +
                           "'");
}

std::string get_element_symbol(int atomicNumber) {
  int num = static_cast<int>(num_elements_defined());
  if (atomicNumber > num || atomicNumber < 1) {
    throw std::runtime_error(
        "At getElementSymbol: AtomicWeightTable index out of bounds: " +
        std::to_string(atomicNumber) +
        " while num_elements_defined() = " + std::to_string(num));
  }
  return AtomicWeightTable[atomicNumber - 1].symbol;
}

std::string get_element_name(const std::string& ename) {
  std::string symbol = trim_copy(ename);
  for (const auto& atom : AtomicWeightTable) {
    if (symbol == atom.symbol) {
      return atom.fullName;
    }
  }
  for (const auto& atom : IsotopeWeightTable) {
    if (symbol == atom.symbol) {
      return atom.fullName;
    }
  }
  throw std::runtime_error("At getElementName: element not found: '" + ename +
                           "'");
}

std::string get_element_name(int atomicNumber) {
  int num = static_cast<int>(num_elements_defined());
  if (atomicNumber > num || atomicNumber < 1) {
    throw std::runtime_error(
        "At getElementName: AtomicWeightTable index out of bounds: " +
        std::to_string(atomicNumber) +
        " while num_elements_defined() = " + std::to_string(num));
  }
  return AtomicWeightTable[atomicNumber - 1].fullName;
}

int get_atomic_number(const std::string& ename) {
  size_t numElements = num_elements_defined();
  size_t numIsotopes = num_isotopes_defined();
  std::string symbol = trim_copy(ename);
  std::string name = to_lower_copy(symbol);
  for (size_t i = 0; i < numElements; i++) {
    if (symbol == AtomicWeightTable[i].symbol) {
      return static_cast<int>(i) + 1;
    } else if (name == AtomicWeightTable[i].fullName) {
      return static_cast<int>(i) + 1;
    }
  }
  for (size_t i = 0; i < numIsotopes; i++) {
    if (symbol == IsotopeWeightTable[i].symbol) {
      return IsotopeWeightTable[i].atomicNumber;
    } else if (name == IsotopeWeightTable[i].fullName) {
      return IsotopeWeightTable[i].atomicNumber;
    }
  }
  throw std::runtime_error("At getAtomicNumber: element not found: " + ename);
}

size_t num_elements_defined() { return AtomicWeightTable.size(); }

size_t num_isotopes_defined() { return IsotopeWeightTable.size(); }

}  // namespace harp
