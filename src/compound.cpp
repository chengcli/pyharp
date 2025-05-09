// C/C++
#include <cctype>
#include <stack>

// harp
#include "compound.hpp"
#include "element.hpp"

namespace harp {

Composition get_composition(const std::string& formula) {
  std::map<std::string, double> elementCount;
  std::stack<std::map<std::string, double>> groupStack;
  std::stack<int> multiplierStack;

  size_t i = 0;
  while (i < formula.size()) {
    if (std::isupper(formula[i])) {
      // Parse element symbol
      std::string element(1, formula[i]);
      i++;
      if (i < formula.size() && std::islower(formula[i])) {
        element += formula[i];
        i++;
      }

      // Parse number (optional)
      std::string numStr;
      while (i < formula.size() && std::isdigit(formula[i])) {
        numStr += formula[i];
        i++;
      }
      double count = numStr.empty() ? 1 : std::stod(numStr);

      // Add to the map
      elementCount[element] += count;
    } else if (formula[i] == '(') {
      // Start a new group
      groupStack.push(elementCount);
      elementCount.clear();
      multiplierStack.push(1);
      i++;
    } else if (formula[i] == ')') {
      // End of a group, get multiplier
      i++;
      std::string numStr;
      while (i < formula.size() && std::isdigit(formula[i])) {
        numStr += formula[i];
        i++;
      }
      int multiplier = numStr.empty() ? 1 : std::stoi(numStr);

      // Apply multiplier to the current group
      for (auto& [element, count] : elementCount) {
        count *= multiplier;
      }

      // Merge back into the previous group
      if (!groupStack.empty()) {
        auto prevGroup = groupStack.top();
        groupStack.pop();
        for (auto& [element, count] : elementCount) {
          prevGroup[element] += count;
        }
        elementCount = prevGroup;
      }
    }
  }

  return elementCount;
}

double get_compound_weight(const Composition& composition) {
  double weight = 0;
  for (const auto& [element, count] : composition) {
    weight += count * get_element_weight(element);
  }
  return weight * 1.e-3;  // kg/kmol -> kg/mol
}

}  // namespace harp
