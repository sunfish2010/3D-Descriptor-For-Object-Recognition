
#include "utilityCore.hpp"


std::string  utilityCore::getFilePathExtension(const std::string &fn) {
    if (fn.find_last_of(".") != std::string::npos)
        return fn.substr(fn.find_last_of(".") + 1);
    return "";
}