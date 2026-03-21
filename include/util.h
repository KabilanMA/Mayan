#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>

// ─── Debug Logger ────────────────────────────────────────────────────────────
// Usage: mayan_debug("My variable is: " << my_var);
// Enable by compiling with: g++ -DMAYAN_ENABLE_DEBUG ...
#ifdef MAYAN_ENABLE_DEBUG
#define mayan_debug(msg) do { std::cerr << "[DEBUG " << __FILE__ << ":" << __LINE__ << "] " << msg << std::endl; } while(0)
#else
#define mayan_debug(msg) do {} while(0)
#endif

// Generic stream insertion operator for std::vector
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i < v.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

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