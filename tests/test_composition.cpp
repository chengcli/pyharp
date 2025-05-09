// C/C++
#include <iostream>

// harp
#include <harp/compound.hpp>

int main() {
  std::string formula = "C6H12O6";

  harp::Composition result = harp::get_composition(formula);

  std::cout << "Parsed Elements:\n";
  for (const auto& [element, count] : result) {
    std::cout << element << ": " << count << '\n';
  }

  double weight = harp::get_compound_weight(result);
  std::cout << "Molecular Weight: " << weight << '\n';

  return 0;
}
