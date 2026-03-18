#include <iostream>
#include <vector>

template <typename Container>
void kPrint(const Container& v) {
    std::cout << "[";
    auto it = v.begin();
    while (it != v.end()) {
        std::cout << *it;
        ++it;
        if (it != v.end()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

template <typename Container>
void kPrintln(const Container& v) {
    kPrint(v);
    std::cout << "\n";
}

template <typename K, typename V>
void kPrint(const std::unordered_map<K, std::vector<V>>& m) {
    std::cout << "{\n";
    bool first = true;

    for (const auto& [key, vec] : m) {
        if (!first) {
            std::cout << ",\n";
        }
        std::cout << "  " << key << ": ";
        kPrint(vec);
        first = false;
    }

    std::cout << "\n}";
}

template <typename K, typename V>
void kPrintln(const std::unordered_map<K, std::vector<V>>& m) {
    kPrint(m);
    std::cout << "\n";
}