#include <iostream>
#include <vector>

void kPrint(const std::vector<char>& v) {
    for (char c : v) {
        std::cout << c << " ";
    }
    std::cout << std::endl;
}